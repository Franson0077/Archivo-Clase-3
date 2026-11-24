"""
prueba_rag.py - Pruebas pedidas por la consigna, contra la API real (Pre-entrega 3).

Uso:
    python prueba_rag.py                    -> corre las 2 pruebas
    python prueba_rag.py "tu pregunta"      -> hace una pregunta libre
    python prueba_rag.py --debug            -> logs detallados

Pruebas:
    1) PREGUNTA CON RESPUESTA: la respuesta esta en data/cadena_de_frio.md.
       Se verifica que responda con el dato correcto y cite ese archivo.
    2) PREGUNTA TRAMPA: un tema que NO esta en ningun documento (descuentos a
       obras sociales). Se verifica que responda exactamente "No lo sé." y sin
       referencias, es decir, que NO alucine.

Si la base vectorial no existe, get_rag_response() la crea antes (ingesta.py).
"""

import asyncio
import json
import logging
import sys

from rag import RAGError, get_rag_response
from schemas import NO_LO_SE, RespuestaRAG

PRUEBAS = [
    {
        "nombre": "1) Pregunta cuya respuesta SI esta en los documentos",
        "pregunta": (
            "Si detecto una excursión de temperatura en un producto refrigerado, "
            "¿en cuánto tiempo tengo que avisar al Director Técnico?"
        ),
        "en_contexto": True,
        "debe_contener": "2 horas",
        "archivo_esperado": "cadena_de_frio.md",
    },
    {
        "nombre": "2) Pregunta TRAMPA (la respuesta NO esta en los documentos)",
        "pregunta": (
            "¿Qué porcentaje de descuento se les hace a las obras sociales en los "
            "medicamentos oncológicos?"
        ),
        "en_contexto": False,
    },
]


def configurar_logs(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    nivel = logging.DEBUG if debug else logging.INFO
    for nombre in ("rag", "pipeline"):  # "pipeline" = logs por intento (chain.py)
        logging.getLogger(nombre).setLevel(nivel)


def mostrar(respuesta: RespuestaRAG) -> None:
    print(json.dumps(respuesta.model_dump(mode="json"), ensure_ascii=False, indent=2), flush=True)


def verificar(prueba: dict, respuesta: RespuestaRAG) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacia = la prueba paso)."""
    problemas = []
    if respuesta.en_contexto != prueba["en_contexto"]:
        problemas.append(f"en_contexto={respuesta.en_contexto}, se esperaba {prueba['en_contexto']}")
    if prueba["en_contexto"]:
        if prueba["debe_contener"] not in respuesta.respuesta:
            problemas.append(f"la respuesta no menciona {prueba['debe_contener']!r}")
        archivos = {r.archivo for r in respuesta.referencias}
        if prueba["archivo_esperado"] not in archivos:
            problemas.append(f"no cita {prueba['archivo_esperado']} (cita: {sorted(archivos)})")
    else:
        if respuesta.respuesta != NO_LO_SE:
            problemas.append(f"se esperaba exactamente {NO_LO_SE!r} (posible alucinacion)")
        if respuesta.referencias:
            problemas.append("una respuesta 'No lo sé' no deberia tener referencias")
    return problemas


async def correr_pruebas() -> int:
    fallidas = 0
    for prueba in PRUEBAS:
        print(f"\n=== {prueba['nombre']} ===", flush=True)
        try:
            respuesta = await get_rag_response(prueba["pregunta"])
        except RAGError as exc:
            print(f"[ERROR CONTROLADO] {exc}")
            fallidas += 1
            continue
        mostrar(respuesta)
        problemas = verificar(prueba, respuesta)
        if problemas:
            fallidas += 1
            print("RESULTADO: FALLO -> " + "; ".join(problemas))
        else:
            print("RESULTADO: PASO")
    print(f"\n{len(PRUEBAS) - fallidas}/{len(PRUEBAS)} pruebas pasaron.")
    return 1 if fallidas else 0


async def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--debug"]
    configurar_logs(debug="--debug" in sys.argv)
    try:
        if args:
            respuesta = await get_rag_response(" ".join(args))
            mostrar(respuesta)
            return 0
        return await correr_pruebas()
    except RAGError as exc:
        print(f"[ERROR CONTROLADO] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Pregunta vacia, configuracion invalida (.env) o indice incompatible.
        print(f"Error de entrada o configuracion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
