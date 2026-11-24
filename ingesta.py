"""
ingesta.py - Modulo de ingesta del RAG (Pre-entrega 3): documentos -> fragmentos -> ChromaDB.

Uso:
    python ingesta.py            -> indexa data/ solo si algo cambio
    python ingesta.py --forzar   -> borra el indice y vuelve a indexar todo

Flujo:
    data/*.md, *.txt  --cargar_documentos-->  Document (1 por archivo)
                      --fragmentar-------->   fragmentos de 500 tokens, 50 de solapamiento
                      --embeddings-------->   vectores (OpenAI text-embedding-3-small)
                      --ChromaDB---------->   coleccion persistente en ./vectorstore

Mapa de la consigna dentro de este archivo:
    "Script de carga que lea archivos de /data"     -> cargar_documentos()
    "Chunking estrategico (>= 500 tokens, 50 overlap)" -> fragmentar()
                                                     (RecursiveCharacterTextSplitter)
    "Cliente persistente en ./vectorstore"           -> abrir_cliente()
    "Verificar si la base ya existe antes de re-indexar" -> indexar() + huella_indice()
    "Mismo modelo de embeddings para indexar y consultar" -> se guarda en la metadata
                                                     de la coleccion y abrir_vectorstore()
                                                     lo comprueba.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reutilizacion del Modulo 1: importar llm_manager ejecuta load_dotenv() y trae
# el lector de numeros del .env; providers.py crea los embeddings con la clave.
from llm_manager import _env_number
from providers import PROVIDERS

log = logging.getLogger("rag.ingesta")

# =============================================================================
# Configuracion (todo se puede cambiar desde .env)
# =============================================================================
# Las rutas se calculan desde la carpeta de ESTE archivo, asi el script funciona
# aunque se ejecute desde otra carpeta.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
VECTORSTORE_DIR = Path(os.getenv("RAG_VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
COLECCION = "documentos"
EXTENSIONES = (".md", ".txt")

# Chunking: la consigna pide fragmentos de MINIMO 500 tokens con 50 de solapamiento.
# Se miden en TOKENS (no en caracteres) con el mismo tokenizador de OpenAI.
CHUNK_TOKENS = _env_number("RAG_CHUNK_TOKENS", 500, int)
CHUNK_OVERLAP = _env_number("RAG_CHUNK_OVERLAP", 50, int)
TOKENIZADOR = "cl100k_base"

# Error comun #1 de la consigna: indexar con un modelo de embeddings y consultar
# con otro. Por eso el nombre del modelo se guarda junto al indice.
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")


# =============================================================================
# Errores propios
# =============================================================================
class IndiceNoEncontradoError(Exception):
    """Todavia no se ejecuto la ingesta."""


class EmbeddingsIncompatiblesError(Exception):
    """El indice se creo con un modelo de embeddings distinto al que se quiere usar."""


@dataclass
class ResultadoIngesta:
    indexado: bool        # True si se (re)indexo; False si el indice ya estaba al dia
    archivos: int
    fragmentos: int
    motivo: str


# =============================================================================
# Embeddings y cliente de ChromaDB
# =============================================================================
def crear_embeddings(model: str = EMBEDDINGS_MODEL) -> Embeddings:
    """OpenAIEmbeddings con la misma clave, reintentos y timeout del Modulo 1."""
    proveedor = PROVIDERS["openai"](
        model=None,
        max_retries=_env_number("LLM_MAX_RETRIES", 2, int),
        timeout=_env_number("LLM_TIMEOUT", 60.0, float),
    )
    return proveedor.create_embeddings(model)


def abrir_cliente(vectorstore_dir: Path = VECTORSTORE_DIR) -> chromadb.ClientAPI:
    """Cliente PERSISTENTE: la base se guarda en disco y sobrevive entre ejecuciones."""
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    # anonymized_telemetry=False: Chroma no envia estadisticas de uso a internet.
    return chromadb.PersistentClient(
        path=str(vectorstore_dir), settings=Settings(anonymized_telemetry=False)
    )


def _coleccion_existente(cliente: chromadb.ClientAPI):
    """Devuelve la coleccion si existe, o None (sin crearla)."""
    nombres = [c if isinstance(c, str) else c.name for c in cliente.list_collections()]
    return cliente.get_collection(COLECCION) if COLECCION in nombres else None


# =============================================================================
# Carga y chunking
# =============================================================================
def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee todos los .md y .txt de data_dir (un Document por archivo)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {data_dir}")

    documentos = []
    for ruta in sorted(data_dir.iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        texto = ruta.read_text(encoding="utf-8").strip()
        if not texto:
            log.warning("Se ignora %s: esta vacio.", ruta.name)
            continue
        documentos.append(Document(page_content=texto, metadata={"archivo": ruta.name}))

    if not documentos:
        raise FileNotFoundError(f"No hay archivos {EXTENSIONES} con texto en {data_dir}")
    return documentos


def crear_splitter(
    chunk_tokens: int = CHUNK_TOKENS, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter que mide en tokens (Unidad 2 del curso).

    "Recursive": intenta cortar primero por parrafos ("\\n\\n"), despues por
    lineas, despues por palabras. Asi los fragmentos respetan la estructura del
    texto y no se cortan frases a la mitad, salvo que sea inevitable.
    """
    if chunk_tokens < 500:
        raise ValueError("RAG_CHUNK_TOKENS debe ser >= 500 (lo pide la consigna).")
    if not 50 <= chunk_overlap < chunk_tokens:
        raise ValueError("RAG_CHUNK_OVERLAP debe ser >= 50 y menor que RAG_CHUNK_TOKENS.")
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZADOR,
        chunk_size=chunk_tokens,
        chunk_overlap=chunk_overlap,
        # Separadores pensados para Markdown: titulos, parrafos, lineas, palabras.
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        add_start_index=True,  # guarda en que caracter del archivo empieza cada fragmento
    )


def fragmentar(documentos: list[Document], splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    """Divide cada documento y numera los fragmentos: id = "archivo#n"."""
    splitter = splitter or crear_splitter()
    fragmentos: list[Document] = []
    for documento in documentos:
        for numero, fragmento in enumerate(splitter.split_documents([documento])):
            fragmento.metadata["fragmento"] = numero
            fragmento.id = f"{documento.metadata['archivo']}#{numero}"
            fragmentos.append(fragmento)
    return fragmentos


# =============================================================================
# Persistencia: indexar solo si hace falta
# =============================================================================
def huella_indice(documentos: list[Document], embeddings_model: str, chunk_tokens: int, chunk_overlap: int) -> str:
    """"Huella digital" de lo que se va a indexar.

    Si cambia CUALQUIER cosa que afecte al indice (el texto de un archivo, un
    archivo nuevo o borrado, el tamano de fragmento o el modelo de embeddings),
    la huella cambia y hay que volver a indexar. Si no cambio nada, se evita
    llamar de nuevo a la API de embeddings (tiempo y dinero).
    """
    h = hashlib.sha256()
    h.update(f"{embeddings_model}|{TOKENIZADOR}|{chunk_tokens}|{chunk_overlap}".encode())
    for documento in documentos:
        h.update(documento.metadata["archivo"].encode())
        h.update(documento.page_content.encode())
    return h.hexdigest()


def indexar(
    forzar: bool = False,
    *,
    data_dir: Path = DATA_DIR,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    embeddings: Embeddings | None = None,
    embeddings_model: str = EMBEDDINGS_MODEL,
    chunk_tokens: int = CHUNK_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> ResultadoIngesta:
    """Carga, fragmenta e indexa data/ en ChromaDB, SOLO si hace falta.

    Los parametros con * permiten usar otra carpeta o embeddings falsos en los
    tests (test_rag.py) sin tocar la base real ni gastar tokens.
    """
    documentos = cargar_documentos(data_dir)
    splitter = crear_splitter(chunk_tokens, chunk_overlap)
    fragmentos = fragmentar(documentos, splitter)
    huella = huella_indice(documentos, embeddings_model, chunk_tokens, chunk_overlap)

    cliente = abrir_cliente(vectorstore_dir)
    existente = _coleccion_existente(cliente)

    # --- Verificar si la base ya existe y esta al dia ---------------------
    if existente is not None and not forzar:
        metadata = existente.metadata or {}
        if metadata.get("huella") == huella and existente.count() == len(fragmentos):
            log.info(
                "Indice al dia (%d fragmentos, embeddings=%s): no se re-indexa.",
                existente.count(), embeddings_model,
            )
            return ResultadoIngesta(False, len(documentos), len(fragmentos), "sin cambios")

    if existente is None:
        motivo = "no existia el indice"
    elif forzar:
        motivo = "re-indexado forzado"
    else:
        motivo = "cambiaron los documentos o la configuracion"
    if existente is not None:
        # Se borra la coleccion entera: asi no quedan fragmentos viejos de un
        # archivo que se modifico o se elimino.
        cliente.delete_collection(COLECCION)
    log.info("Indexando: %s.", motivo)

    inicio = time.perf_counter()
    vectorstore = Chroma(
        client=cliente,
        collection_name=COLECCION,
        embedding_function=embeddings or crear_embeddings(embeddings_model),
        # Se guarda con el indice: que modelo de embeddings se uso y la huella.
        collection_metadata={
            "huella": huella,
            "embeddings_model": embeddings_model,
            "chunk_tokens": chunk_tokens,
            "chunk_overlap": chunk_overlap,
        },
        # Distancia coseno: la habitual para embeddings de texto.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    vectorstore.add_documents(fragmentos, ids=[f.id for f in fragmentos])

    for documento in documentos:
        n = sum(f.metadata["archivo"] == documento.metadata["archivo"] for f in fragmentos)
        log.info("  %s -> %d fragmento(s)", documento.metadata["archivo"], n)
    log.info(
        "Indexados %d fragmentos de %d archivos en %.1f s (embeddings=%s).",
        len(fragmentos), len(documentos), time.perf_counter() - inicio, embeddings_model,
    )
    return ResultadoIngesta(True, len(documentos), len(fragmentos), motivo)


def abrir_vectorstore(
    *,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    embeddings: Embeddings | None = None,
    embeddings_model: str = EMBEDDINGS_MODEL,
) -> Chroma:
    """Abre el indice YA creado para consultarlo.

    Comprueba que se consulte con el MISMO modelo de embeddings con el que se
    indexo. Si no coincide, los vectores no son comparables y los resultados
    serian aleatorios (error #1 de la consigna): se corta con un error claro.
    """
    cliente = abrir_cliente(vectorstore_dir)
    coleccion = _coleccion_existente(cliente)
    if coleccion is None or coleccion.count() == 0:
        raise IndiceNoEncontradoError(
            f"No hay indice en {vectorstore_dir}. Ejecuta primero: python ingesta.py"
        )
    modelo_indice = (coleccion.metadata or {}).get("embeddings_model")
    if modelo_indice != embeddings_model:
        raise EmbeddingsIncompatiblesError(
            f"El indice se creo con embeddings {modelo_indice!r} y se quiere consultar con "
            f"{embeddings_model!r}. Re-indexa con: python ingesta.py --forzar"
        )
    return Chroma(
        client=cliente,
        collection_name=COLECCION,
        embedding_function=embeddings or crear_embeddings(embeddings_model),
        create_collection_if_not_exists=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa los documentos de data/ en ChromaDB.")
    parser.add_argument("--forzar", action="store_true", help="re-indexar aunque no haya cambios")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    # Las librerias HTTP loguean cada peticion en INFO; aca no aporta.
    for ruidoso in ("httpx", "httpx2"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    try:
        resultado = indexar(forzar=args.forzar)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    estado = "indexado" if resultado.indexado else "sin cambios"
    print(
        f"Listo ({estado}): {resultado.archivos} archivos, {resultado.fragmentos} fragmentos "
        f"en {VECTORSTORE_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
