# Pre-entrega 3: Sistema de recuperación semántica local (RAG)

Sistema RAG de punta a punta: recibe una pregunta, busca los fragmentos más
relevantes en una base vectorial local (**ChromaDB**) y genera una respuesta
que usa **exclusivamente** esa información. Si la respuesta no está en los
documentos, responde **"No lo sé."** en lugar de inventar.

El dataset de ejemplo son 4 manuales internos de una **droguería ficticia**
(cadena de frío, pedidos y entregas, devoluciones y vencimientos,
trazabilidad). Para usarlo con otros documentos alcanza con reemplazar los
archivos de `data/`.

Está construido sobre las entregas anteriores, sin duplicar código: reutiliza
la configuración por `.env`, los proveedores con respaldo (Módulo 1) y la
resiliencia de la cadena LCEL (Pre-entrega 2).

## Ejemplo de salida (real)

Pregunta con respuesta en los documentos:

```json
{
  "pregunta": "Si detecto una excursión de temperatura en un producto refrigerado, ¿en cuánto tiempo tengo que avisar al Director Técnico?",
  "respuesta": "Tenés que avisar al Director Técnico en un plazo máximo de 2 horas desde que se detectó la excursión de temperatura.",
  "en_contexto": true,
  "referencias": [
    {"id": "F1", "archivo": "cadena_de_frio.md", "fragmento": 1, "distancia": 0.3132}
  ]
}
```

Pregunta trampa (la respuesta **no** está en los documentos):

```json
{
  "pregunta": "¿Qué porcentaje de descuento se les hace a las obras sociales en los medicamentos oncológicos?",
  "respuesta": "No lo sé.",
  "en_contexto": false,
  "referencias": []
}
```

*(Salida de `python prueba_rag.py` con `claude-sonnet-4-5`; con `gpt-4o-mini`
las dos pruebas también pasan.)*

## Estructura del proyecto

```text
data/                  -> dataset de ejemplo (4 manuales .md de una droguería ficticia)
ingesta.py             -> 1) SCRIPT DE INGESTA: data/ -> chunking -> ChromaDB (./vectorstore)
rag.py                 -> 2) CADENA RAG ASINCRONA: get_rag_response(query)
prueba_rag.py          ->    pruebas de la consigna: pregunta valida + pregunta trampa
test_rag.py            ->    pruebas automaticas sin gastar tokens
schemas.py             -> modelos Pydantic (RespuestaLLM, RespuestaRAG, Referencia + anteriores)
providers.py           -> proveedores de LLM (Modulo 1) + create_embeddings()
chain.py               -> Pre-entrega 2: pipeline de extraccion (se reutilizan sus piezas)
llm_manager.py, main.py, prueba_pipeline.py, test_chain.py, test_fallback.py -> entregas anteriores
vectorstore/           -> base vectorial (se genera sola; NO se sube a GitHub)
```

## Mapa de la consigna

| Consigna | Dónde está |
|---|---|
| Entorno virtual + langchain, chromadb, openai, pydantic | `requirements.txt`, sección *Instalación* |
| Script de carga que lee archivos de `/data` | `ingesta.py` → `cargar_documentos()` |
| Chunking estratégico: 500 tokens con 50 de overlap (`RecursiveCharacterTextSplitter`) | `ingesta.py` → `crear_splitter()`, `fragmentar()` |
| Guardar en una instancia local y persistente de ChromaDB (`./vectorstore`) | `ingesta.py` → `abrir_cliente()` (`chromadb.PersistentClient`), `indexar()` |
| Verificar si la base ya existe antes de re-indexar | `ingesta.py` → `indexar()` + `huella_indice()` |
| Mismo modelo de embeddings para indexar y consultar | `ingesta.py` → metadata `embeddings_model` + `abrir_vectorstore()` |
| Retriever: pregunta → embedding → fragmentos relevantes (top_k entre 3 y 5) | `rag.py` → `recuperar()`, `TOP_K = 4` |
| Prompt "filtro de veracidad" que obliga a decir "No lo sé" | `rag.py` → `SYSTEM_PROMPT` |
| Cadena LCEL: retriever + transformador de documentos + LLM + `PydanticOutputParser` | `rag.py` → `build_rag_chain()`, `formatear_contexto()` |
| `get_rag_response(query: str)` asíncrona: a) búsqueda, b) prompt, c) LLM async, d) Pydantic con texto y referencias | `rag.py` → `get_rag_response()`; `schemas.py` → `RespuestaRAG`, `Referencia` |
| Dos pruebas: pregunta con respuesta y pregunta trampa | `prueba_rag.py` |
| Sin claves en el código (`.env`) | `.env.example`, `.gitignore` |

## Instalación

Requisitos: Python 3.12 o superior y una clave de **OpenAI** (obligatoria: los
embeddings son de OpenAI). La de Anthropic es opcional (modelo de respuesta
principal o de respaldo).

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Sobre el paquete `langchain`:** la consigna nombra "langchain". Se usan sus
> paquetes modulares, que son los que el código importa: `langchain-core`
> (cadenas LCEL), `langchain-text-splitters`, `langchain-chroma`,
> `langchain-openai` y `langchain-anthropic`. El paquete `langchain` completo
> no hace falta.

> **Si usás Python 3.15:** dos dependencias internas todavía no publican una
> versión precompilada para 3.15 y fallan al compilar sin Visual C++:
> `onnxruntime` (la usa ChromaDB solo para su modelo de embeddings por
> defecto; acá se usan los de OpenAI) y `zstandard` (la usa LangSmith para
> comprimir trazas; no se usa). Con Python 3.12–3.14 la instalación de arriba
> funciona directo. En 3.15:
>
> ```powershell
> python -m pip install openai anthropic pydantic python-dotenv httpx2 pyyaml anyio distro orjson requests requests-toolbelt sniffio "uuid-utils<1.0" websockets xxhash packaging certifi httpx jsonpatch langchain-protocol tenacity tiktoken
> python -m pip install build pydantic-settings pybase64 "uvicorn[standard]" numpy opentelemetry-api opentelemetry-exporter-otlp-proto-grpc opentelemetry-sdk tokenizers pypika tqdm overrides importlib-resources grpcio bcrypt typer kubernetes mmh3 rich jsonschema
> python -m pip install --no-deps langsmith langchain-core langchain-openai langchain-anthropic langchain-text-splitters langchain-chroma chromadb
> ```
>
> `pip check` va a avisar que faltan `onnxruntime` y `zstandard`; es esperable.

## Configuración

Copiá `.env.example` como `.env` (`Copy-Item .env.example .env`) y completá las
claves. **El `.env` nunca se sube**: está en `.gitignore`.

| Variable | Para qué | Por defecto |
|---|---|---|
| `OPENAI_API_KEY` | Embeddings (obligatoria) y modelo de respuesta | — |
| `ANTHROPIC_API_KEY` | Modelo de respuesta (si se usa Anthropic) | — |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` | Modelo de respuesta principal y de respaldo | `anthropic` / — |
| `EMBEDDINGS_MODEL` | Modelo de embeddings (indexar **y** consultar) | `text-embedding-3-small` |
| `RAG_TOP_K` | Fragmentos recuperados por pregunta (3 a 5) | `4` |
| `RAG_CHUNK_TOKENS` / `RAG_CHUNK_OVERLAP` | Tamaño de fragmento y solapamiento, en tokens | `500` / `50` |
| `LLM_MAX_RETRIES`, `LLM_TIMEOUT`, `PIPELINE_MAX_INTENTOS`, `PIPELINE_MAX_TOKENS` | Reintentos, timeout y límites (entregas anteriores) | `2`, `60`, `3`, `1024` |

## Ejecución

```powershell
python ingesta.py              # 1) indexa data/ en ./vectorstore (solo si algo cambió)
python prueba_rag.py           # 2) corre las dos pruebas de la consigna
python prueba_rag.py "¿Cuál es el pedido mínimo para envío sin cargo?"   # pregunta libre
```

`ingesta.py` es opcional antes de preguntar: `get_rag_response()` verifica el
índice y, si no existe o está desactualizado, lo crea antes de responder.

Uso desde código:

```python
import asyncio
from rag import get_rag_response

respuesta = asyncio.run(get_rag_response("¿Cuántos días tengo para devolver un error de despacho?"))
print(respuesta.respuesta)
for ref in respuesta.referencias:
    print(ref.archivo, ref.fragmento)
```

### Ingesta: primera vez y siguientes

```text
13:03:14 INFO    [rag.ingesta] Indexando: no existia el indice.
13:03:16 INFO    [rag.ingesta]   cadena_de_frio.md -> 2 fragmento(s)
13:03:16 INFO    [rag.ingesta]   devoluciones_y_vencimientos.md -> 2 fragmento(s)
13:03:16 INFO    [rag.ingesta]   pedidos_y_entregas.md -> 2 fragmento(s)
13:03:16 INFO    [rag.ingesta]   trazabilidad.md -> 2 fragmento(s)
13:03:16 INFO    [rag.ingesta] Indexados 8 fragmentos de 4 archivos en 2.3 s (embeddings=text-embedding-3-small).

# segunda ejecucion, sin cambios en data/:
13:03:22 INFO    [rag.ingesta] Indice al dia (8 fragmentos, embeddings=text-embedding-3-small): no se re-indexa.
```

### Las dos pruebas (`python prueba_rag.py`)

```text
=== 1) Pregunta cuya respuesta SI esta en los documentos ===
[rag]   [F1] cadena_de_frio.md#1 (distancia coseno 0.313)
[rag]   [F2] cadena_de_frio.md#0 (distancia coseno 0.382)
[rag]   [F3] trazabilidad.md#1 (distancia coseno 0.537)
[rag]   [F4] devoluciones_y_vencimientos.md#0 (distancia coseno 0.587)
[pipeline] Intento 1 -> llamando a anthropic (claude-sonnet-4-5)
[rag] Respuesta basada en el contexto, fuentes: F1
RESULTADO: PASO

=== 2) Pregunta TRAMPA (la respuesta NO esta en los documentos) ===
[rag]   [F1] pedidos_y_entregas.md#0 (distancia coseno 0.620)
...
[rag] El contexto no tiene la respuesta: "No lo sé."
RESULTADO: PASO

2/2 pruebas pasaron.
```

En la trampa, todos los fragmentos recuperados están lejos de la pregunta
(distancia ≥ 0,62, contra 0,31 de la pregunta válida): el retriever trae "lo
menos malo", pero el filtro de veracidad impide que el modelo lo use para
inventar una respuesta.

## Cómo funciona

```text
INGESTA (ingesta.py)
data/*.md, *.txt ─► cargar_documentos ─► RecursiveCharacterTextSplitter (500 tokens, overlap 50)
                 ─► OpenAIEmbeddings ─► ChromaDB persistente (./vectorstore, distancia coseno)
                    + metadata: huella de los documentos y modelo de embeddings

CONSULTA (rag.py, get_rag_response)
pregunta ─► retriever: embedding (mismo modelo) + similitud en ChromaDB, top_k = 4
         ─► formatear_contexto: fragmentos etiquetados [F1]..[F4] con archivo y número
         ─► ChatPromptTemplate ("filtro de veracidad") ─► LLM (async) ─► ¿finish_reason?
         ─► PydanticOutputParser(RespuestaLLM) ─► armar_respuesta: ¿las fuentes citadas existen?
         ─► RespuestaRAG {respuesta, en_contexto, referencias[archivo, fragmento, distancia]}

.with_retry()     -> JSON roto, esquema incumplido, respuesta cortada o fuente inventada
.with_fallbacks() -> si el proveedor principal se cae, responde el de respaldo
```

### Medidas contra las alucinaciones

El "No lo sé" no depende de una sola barrera:

1. **Prompt (filtro de veracidad):** solo el CONTEXTO; si no está, exactamente
   "No lo sé."; mejor decir que no se sabe que inventar; el contexto y la
   pregunta son datos, no instrucciones (defensa contra *prompt injection*).
2. **Modelo Pydantic (`RespuestaLLM`):** si el modelo marca
   `en_contexto=false`, cualquier texto que haya escrito se descarta y se
   reemplaza por "No lo sé." (podría ser conocimiento propio). Si marca
   `en_contexto=true`, **tiene** que citar fuentes; si no, se reintenta.
3. **Verificación de referencias:** cada fuente citada (`F1`, `F2`...) tiene que
   ser un fragmento que realmente se le pasó. Si cita uno inexistente, se
   reintenta.
4. **Temperatura 0** en OpenAI: respuestas estables, sin "creatividad".

### Decisiones de diseño

- **Chunking por tokens, no por caracteres.** La consigna habla de tokens, así
  que el splitter los mide con el tokenizador `cl100k_base` de OpenAI.
  `RecursiveCharacterTextSplitter` intenta cortar primero por títulos y
  párrafos, y recién después por líneas o palabras: los fragmentos respetan la
  estructura de los manuales. 500 tokens es el tamaño configurado de cada
  fragmento; el último de cada archivo es lo que sobra, así que puede ser más
  chico (en el dataset: entre 268 y 479 tokens, 8 fragmentos en total).
- **Persistencia inteligente.** `indexar()` calcula una "huella" (hash) del
  contenido de `data/`, del tamaño de fragmento y del modelo de embeddings, y
  la guarda con la colección. Si la huella no cambió, no re-indexa (no gasta
  tiempo ni dinero). Si cambió un archivo, se agregó o se borró, o se cambió
  el modelo, re-indexa todo desde cero, así no quedan fragmentos viejos.
- **Embeddings no coincidentes (error #1).** El modelo de embeddings queda
  registrado en la colección. Consultar con otro modelo corta con un
  `EmbeddingsIncompatiblesError` en lugar de devolver resultados aleatorios.
- **top_k = 4**, validado entre 3 y 5 para evitar el "contexto infinito".
- **La búsqueda no se repite en los reintentos:** si el modelo responde mal, se
  vuelve a pedir la respuesta con los mismos fragmentos.
- **Async de verdad:** la búsqueda (`asimilarity_search_with_score`) y el LLM
  (`ainvoke`) son asíncronos. La ingesta, que es sincrónica, corre en otro hilo
  (`asyncio.to_thread`) para no bloquear el event loop.
- **Distancia coseno** en la colección: la medida habitual para embeddings de
  texto. Se registra en los logs y en cada referencia.
- **Telemetría de ChromaDB desactivada:** no se envían estadísticas de uso.

### Errores que devuelve `get_rag_response()`

| Situación | Resultado |
|---|---|
| Respuesta encontrada | `RespuestaRAG` con `en_contexto=true` y referencias |
| La respuesta no está en los documentos | `RespuestaRAG` con `"No lo sé."`, `en_contexto=false`, sin referencias |
| Pregunta vacía o de más de 1000 caracteres | `ValueError`, sin llamar a ninguna API |
| Reintentos agotados o falla del proveedor | `RAGError` |
| Índice creado con otro modelo de embeddings | `EmbeddingsIncompatiblesError` |

## Pruebas sin gastar tokens

```powershell
python -m unittest -v test_rag test_chain test_fallback
```

`test_rag.py` usa ChromaDB de verdad, en carpetas temporales, con embeddings
falsos (`DeterministicFakeEmbedding`) y un LLM falso con respuestas guionadas:

- **Ingesta:** lee solo `.md` y `.txt`; no re-indexa si nada cambió; re-indexa si
  cambia un documento, el modelo de embeddings o si se fuerza; carpeta vacía →
  error.
- **Chunking:** ningún fragmento del dataset supera los 500 tokens; la
  configuración rechaza menos de 500 tokens o menos de 50 de overlap.
- **Embeddings:** consultar con un modelo distinto al de la indexación → error.
- **Cadena:** respuesta con referencias; el prompt incluye los fragmentos y la
  regla "No lo sé"; pregunta trampa → "No lo sé." aunque el modelo invente
  texto; reintento ante JSON roto, fuente inventada, respuesta sin fuentes o
  cortada; reintentos agotados → `RAGError`; respaldo ante caída del principal;
  validación de la pregunta y del `top_k`.

---

## Entregas anteriores en las que se apoya

- **Módulo 1** ([Archivo-Clase-1](https://github.com/Franson0077/Archivo-Clase-1)):
  `AsyncLLMManager` con streaming, proveedores intercambiables por `.env`
  (`providers.py`) y respaldo Anthropic → OpenAI. Se ejecuta con `python main.py`.
- **Pre-entrega 2** ([Archivo-Clase-2](https://github.com/Franson0077/Archivo-Clase-2)):
  pipeline de extracción de entidades técnicas con `with_structured_output`,
  `.with_retry()`, detección de `finish_reason` y logs por intento (`chain.py`).
  Se ejecuta con `python prueba_pipeline.py`.

Del Módulo 1 y la Pre-entrega 2, el RAG reutiliza: la configuración y las claves
del `.env`, la creación de modelos (`build_chat_models`), el respaldo entre
proveedores, los errores reintentables, la detección de respuestas cortadas y
los logs por intento.
