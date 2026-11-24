"""
test_rag.py - Pruebas del RAG SIN llamar a las APIs (no gasta tokens ni necesita internet).

Ejecutar:
    python -m unittest -v test_rag

Que se reemplaza:
    - Embeddings: DeterministicFakeEmbedding de LangChain (vectores fijos por texto).
    - LLM: ModeloTexto, un modelo de chat falso con respuestas guionadas (JSON).
    - Carpetas: cada test usa una carpeta temporal, asi la base real no se toca.

Que se prueba de verdad: ChromaDB, el chunking, la persistencia, la cadena LCEL
completa (retriever + prompt + PydanticOutputParser + with_retry + with_fallbacks).
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import tiktoken
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ingesta import (
    TOKENIZADOR,
    EmbeddingsIncompatiblesError,
    IndiceNoEncontradoError,
    abrir_vectorstore,
    cargar_documentos,
    crear_splitter,
    fragmentar,
    indexar,
)
from rag import RAGError, build_rag_chain, formatear_contexto, get_rag_response
from schemas import NO_LO_SE, RespuestaRAG

DATA_REAL = Path(__file__).resolve().parent / "data"
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def json_llm(respuesta: str, en_contexto: bool, fuentes: list[str]) -> str:
    """Texto que "devolveria" el LLM, en el formato que pide PydanticOutputParser."""
    return json.dumps({"respuesta": respuesta, "en_contexto": en_contexto, "fuentes": fuentes})


class ModeloTexto(BaseChatModel):
    """Modelo de chat falso: cada llamada devuelve el siguiente elemento de `guion`.

    str        -> texto de la respuesta
    (str, motivo) -> texto + finish_reason
    Exception  -> la lanza (simula un error de la API)
    """

    guion: list[Any]
    llamadas: int = 0
    ultimo_prompt: str = ""

    @property
    def _llm_type(self) -> str:
        return "falso"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.ultimo_prompt = "\n".join(str(m.content) for m in messages)
        paso = self.guion[min(self.llamadas, len(self.guion) - 1)]
        self.llamadas += 1
        if isinstance(paso, BaseException):
            raise paso
        texto, motivo = paso if isinstance(paso, tuple) else (paso, "end_turn")
        mensaje = AIMessage(content=texto, response_metadata={"stop_reason": motivo})
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


class ConCarpetas(unittest.TestCase):
    """Base: crea data/ y vectorstore/ temporales con 2 documentos chicos."""

    def setUp(self):
        # ignore_cleanup_errors: en Windows, ChromaDB puede dejar archivos abiertos.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        base = Path(self._tmp.name)
        self.data = base / "data"
        self.data.mkdir()
        self.vs = base / "vectorstore"
        (self.data / "frio.md").write_text("Las vacunas se guardan entre 2 y 8 grados.", encoding="utf-8")
        (self.data / "pedidos.txt").write_text("El corte de pedidos en CABA es a las 14 horas.", encoding="utf-8")
        (self.data / "ignorar.pdf").write_text("no se lee", encoding="utf-8")
        self.emb = DeterministicFakeEmbedding(size=32)

    def tearDown(self):
        self._tmp.cleanup()

    def indexar(self, **kwargs):
        kwargs.setdefault("embeddings_model", "fake-a")
        return indexar(data_dir=self.data, vectorstore_dir=self.vs, embeddings=self.emb, **kwargs)


# =============================================================================
# Ingesta
# =============================================================================
class IngestaTests(ConCarpetas):

    def test_indexa_solo_md_y_txt(self):
        resultado = self.indexar()
        self.assertTrue(resultado.indexado)
        self.assertEqual((resultado.archivos, resultado.fragmentos), (2, 2))

    def test_no_reindexa_si_no_hubo_cambios(self):
        self.indexar()
        segundo = self.indexar()
        self.assertFalse(segundo.indexado)
        self.assertEqual(segundo.motivo, "sin cambios")

    def test_reindexa_si_cambia_un_documento_o_se_fuerza(self):
        self.indexar()
        (self.data / "frio.md").write_text("Texto nuevo sobre la cadena de frío.", encoding="utf-8")
        self.assertTrue(self.indexar().indexado)
        self.assertTrue(self.indexar(forzar=True).indexado)

    def test_reindexa_si_cambia_el_modelo_de_embeddings(self):
        self.indexar(embeddings_model="fake-a")
        self.assertTrue(self.indexar(embeddings_model="fake-b").indexado)

    def test_consultar_con_otros_embeddings_es_un_error(self):
        # Error #1 de la consigna: indexar con un modelo y consultar con otro.
        self.indexar(embeddings_model="fake-a")
        with self.assertRaises(EmbeddingsIncompatiblesError):
            abrir_vectorstore(vectorstore_dir=self.vs, embeddings=self.emb, embeddings_model="fake-b")

    def test_sin_indice_avisa_que_hay_que_ejecutar_la_ingesta(self):
        with self.assertRaises(IndiceNoEncontradoError):
            abrir_vectorstore(vectorstore_dir=self.vs, embeddings=self.emb, embeddings_model="fake-a")

    def test_carpeta_sin_documentos(self):
        for archivo in self.data.iterdir():
            archivo.unlink()
        with self.assertRaises(FileNotFoundError):
            self.indexar()


class ChunkingTests(unittest.TestCase):

    def test_fragmentos_de_500_tokens_con_solapamiento(self):
        fragmentos = fragmentar(cargar_documentos(DATA_REAL))
        tokenizador = tiktoken.get_encoding(TOKENIZADOR)
        self.assertGreater(len(fragmentos), len(cargar_documentos(DATA_REAL)))
        for f in fragmentos:
            self.assertLessEqual(len(tokenizador.encode(f.page_content)), 500)
            self.assertEqual(f.id, f"{f.metadata['archivo']}#{f.metadata['fragmento']}")

    def test_la_consigna_exige_minimo_500_y_50(self):
        for tokens, overlap in [(400, 50), (500, 10), (500, 500)]:
            with self.subTest(tokens=tokens, overlap=overlap):
                with self.assertRaises(ValueError):
                    crear_splitter(tokens, overlap)


# =============================================================================
# Cadena RAG
# =============================================================================
class RAGTests(ConCarpetas):

    def setUp(self):
        super().setUp()
        self.indexar()
        self.vectorstore = abrir_vectorstore(
            vectorstore_dir=self.vs, embeddings=self.emb, embeddings_model="fake-a"
        )

    def cadena(self, *modelos, **kwargs):
        kwargs.setdefault("top_k", 3)
        return build_rag_chain(self.vectorstore, list(modelos), espera_exponencial=False, **kwargs)

    def preguntar(self, cadena, pregunta="¿Dónde se guardan las vacunas?") -> RespuestaRAG:
        return asyncio.run(get_rag_response(pregunta, chain=cadena))

    def test_respuesta_con_referencias(self):
        modelo = ModeloTexto(guion=[json_llm("Entre 2 y 8 grados.", True, ["[F1]"])])
        respuesta = self.preguntar(self.cadena(modelo))
        self.assertTrue(respuesta.en_contexto)
        self.assertEqual(respuesta.respuesta, "Entre 2 y 8 grados.")
        self.assertEqual([r.id for r in respuesta.referencias], ["F1"])
        self.assertIn(respuesta.referencias[0].archivo, {"frio.md", "pedidos.txt"})
        self.assertIsNotNone(respuesta.referencias[0].distancia)

    def test_el_prompt_lleva_los_fragmentos_y_la_regla_no_lo_se(self):
        modelo = ModeloTexto(guion=[json_llm(NO_LO_SE, False, [])])
        self.preguntar(self.cadena(modelo))
        self.assertIn("[F1]", modelo.ultimo_prompt)
        self.assertIn("2 y 8 grados", modelo.ultimo_prompt)  # top_k=3 trae ambos documentos
        self.assertIn(f'exactamente "{NO_LO_SE}"', modelo.ultimo_prompt)

    def test_pregunta_trampa_no_alucina(self):
        # Aunque el modelo "invente" texto, si dice en_contexto=false se descarta.
        modelo = ModeloTexto(guion=[json_llm("Un 40% de descuento.", False, ["F1"])])
        respuesta = self.preguntar(self.cadena(modelo), "¿Qué descuento tienen las obras sociales?")
        self.assertFalse(respuesta.en_contexto)
        self.assertEqual(respuesta.respuesta, NO_LO_SE)
        self.assertEqual(respuesta.referencias, [])

    def test_reintenta_ante_salidas_invalidas(self):
        malas = {
            "JSON roto": "esto no es JSON {",
            "fuente inventada": json_llm("Entre 2 y 8 grados.", True, ["F9"]),
            "sin fuentes": json_llm("Entre 2 y 8 grados.", True, []),
            "respuesta cortada": (json_llm("Entre 2 y 8 grados.", True, ["F1"]), "max_tokens"),
        }
        for nombre, mala in malas.items():
            with self.subTest(caso=nombre):
                modelo = ModeloTexto(guion=[mala, json_llm("Entre 2 y 8 grados.", True, ["F1"])])
                respuesta = self.preguntar(self.cadena(modelo))
                self.assertTrue(respuesta.en_contexto)
                self.assertEqual(modelo.llamadas, 2)

    def test_reintentos_agotados_es_error_controlado(self):
        modelo = ModeloTexto(guion=["no es JSON"])
        with self.assertRaises(RAGError):
            self.preguntar(self.cadena(modelo, max_intentos=3))
        self.assertEqual(modelo.llamadas, 3)

    def test_caida_del_principal_usa_el_respaldo(self):
        principal = ModeloTexto(guion=[anthropic.APIConnectionError(request=_REQUEST)])
        respaldo = ModeloTexto(guion=[json_llm("Entre 2 y 8 grados.", True, ["F1"])])
        cadena = self.cadena(principal, respaldo, errores_de_caida=(anthropic.APIConnectionError,))
        self.assertTrue(self.preguntar(cadena).en_contexto)
        self.assertEqual(respaldo.llamadas, 1)

    def test_validaciones_de_entrada(self):
        modelo = ModeloTexto(guion=[json_llm(NO_LO_SE, False, [])])
        for pregunta in ("", "   ", "x" * 1001):
            with self.subTest(largo=len(pregunta)):
                with self.assertRaises(ValueError):
                    self.preguntar(self.cadena(modelo), pregunta)
        self.assertEqual(modelo.llamadas, 0)
        for top_k in (2, 6):  # "contexto infinito": top_k entre 3 y 5
            with self.assertRaises(ValueError):
                self.cadena(modelo, top_k=top_k)

    def test_formatear_contexto_vacio(self):
        self.assertIn("no se encontraron", formatear_contexto([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
