"""
schemas.py - Esquemas de datos del proyecto (modelos Pydantic).

Aca viven SOLO las estructuras de datos: que entra al modelo y que sale
(tanto la respuesta valida como los errores). No hay logica de red ni de
proveedores; por eso este archivo no importa openai, anthropic ni langchain.

Mapa de la consigna dentro de este archivo:
    Modulo 1 / PUNTO 2 -> GenerationParams (validacion de parametros de entrada).
    Modulo 1 / PUNTO 5 -> LLMError (error controlado devuelto como objeto).
    Pre-entrega 2      -> NivelCriticidad + ExtraccionTecnica (contrato de salida
                          del pipeline de extraccion, ver chain.py).
    Pre-entrega 3      -> RespuestaLLM (lo que parsea PydanticOutputParser),
                          Referencia y RespuestaRAG (respuesta final del RAG, ver rag.py).
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Tipos de error posibles. Se define una sola vez y se reutiliza en
# LLMError y en providers.py (classify_error), asi no hay que repetirlo.
ErrorKind = Literal["RATE_LIMIT", "NETWORK", "SERVER", "PROVIDER"]


# =============================================================================
# PUNTO 2 - Esquema Pydantic para validar los parametros de entrada del modelo.
# =============================================================================
class GenerationParams(BaseModel):
    # Pydantic valida los datos cuando se crea GenerationParams(...).
    # Si un dato no cumple, se lanza una ValidationError ANTES de llamar al
    # proveedor: no se gasta una peticion (ni dinero) con datos invalidos.
    #
    # Field(...) declara las reglas de cada campo:
    #   ge = mayor o igual   | gt = mayor estricto
    #   le = menor o igual   | min_length = largo minimo del texto

    # Texto que escribe el usuario. Obligatorio (no tiene default).
    prompt: str = Field(min_length=1, description="Texto enviado al modelo")

    # Instruccion de sistema: define el "rol" o comportamiento del modelo.
    system: str | None = Field(default=None, description="Instruccion de sistema opcional")

    # PUNTO 2 - "temperatura de 0 a 2": ge=0.0 y le=2.0 cumplen exactamente eso.
    # Valores bajos = respuestas mas deterministas; altos = mas creativas.
    # IMPORTANTE: solo la usa OpenAI. El SDK de Anthropic instalado no acepta
    # temperature, pero se valida igual porque si Anthropic se cae, el
    # respaldo (OpenAI) la necesita.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # PUNTO 2 - "max_tokens": limite de largo de la respuesta. Debe ser > 0.
    # Este si lo usan ambos proveedores.
    max_tokens: int = Field(default=500, gt=0, le=100000)

    # Parametro extra (el "etc." de la consigna): nucleus sampling, entre 0 y 1.
    # Igual que temperature: solo lo usa OpenAI.
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)

    # Validadores personalizados: corren DESPUES de las reglas de Field.
    # min_length=1 acepta "   " (tres espacios), por eso hace falta este
    # validador extra que limpia espacios y rechaza el texto vacio.
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El prompt no puede estar vacio.")
        return value

    # Si system llega vacio o con solo espacios, se normaliza a None
    # para que mas adelante no se envie un mensaje de sistema vacio.
    @field_validator("system")
    @classmethod
    def validate_system(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


# =============================================================================
# PUNTO 5 - Error controlado: un OBJETO, no un texto suelto.
# =============================================================================
class LLMError(BaseModel):
    # generate() entrega textos (str) o, si algo falla, UN LLMError.
    # Quien consume pregunta: isinstance(fragmento, LLMError), sin tener que
    # leer el texto para adivinar si es un error.
    #
    # Para un AGENTE esto es clave: puede decidir que hacer segun el tipo de
    # error (esperar, avisar, cambiar de herramienta...) en vez de solo mostrarlo.
    #
    # Alcance: LLMError cubre los errores que vienen de la API. Los errores de
    # configuracion (clave faltante) y los bugs de nuestro codigo NO se
    # convierten en LLMError (ver README, "Que errores se controlan").
    kind: ErrorKind
    provider: str
    message: str

    def __str__(self) -> str:
        # Formato legible para mostrar en la terminal.
        return f"[ERROR:{self.kind}] {self.provider}: {self.message}"


# =============================================================================
# PRE-ENTREGA 2 - Contrato de salida del pipeline de extraccion tecnica.
# =============================================================================
class NivelCriticidad(str, Enum):
    # Enum pedido por la consigna: solo estos tres valores son validos.
    # Hereda de str para que en el JSON aparezca "alta" y no "NivelCriticidad.ALTA".
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


# Valores de relleno que NO cuentan como tecnologia (ver limpiar_tecnologias).
_COMODINES = {
    "indeterminado", "indeterminada", "desconocido", "desconocida", "ninguno",
    "ninguna", "no especificado", "no especificada", "n/a", "na", "none",
    "unknown", "otro", "otros", "-",
}


class ExtraccionTecnica(BaseModel):
    """Entidades tecnicas extraidas de un texto (log de error, descripcion de arquitectura...)."""

    # with_structured_output() convierte esta clase en un JSON Schema y se lo
    # envia al modelo: los "description" de cada campo son INSTRUCCIONES que el
    # LLM lee. Despues, la respuesta se valida con estas mismas reglas.
    #
    # extra="forbid": si el modelo inventa un campo que no existe, es un error.
    model_config = ConfigDict(extra="forbid")

    # Restriccion pedida en "Define tu contrato": la lista no puede estar vacia.
    # A proposito NO se usa Field(min_length=1): eso viajaria al JSON Schema y
    # el modelo, obligado a poner algo, inventaria un relleno ("indeterminado")
    # ante un texto ambiguo. Asi el modelo puede ser honesto (devolver []) y es
    # el validador de abajo el que rechaza la lista vacia -> reintento -> error
    # controlado. Ver la "prueba de estres" en prueba_pipeline.py.
    tecnologias: list[str] = Field(
        description=(
            "Tecnologias, lenguajes, frameworks, bases de datos o servicios "
            "mencionados en el texto. Nombres propios, sin repetir. Lista vacia "
            "si el texto no menciona ninguna."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Gravedad tecnica del problema o riesgo descrito: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        description="Resumen tecnico de una o dos oraciones, en espanol."
    )

    # --- Validadores: corren DESPUES de que el modelo responde ----------------
    # Si alguno falla, with_structured_output reporta un error de parseo y la
    # logica de reintentos de chain.py vuelve a pedirle la respuesta al LLM.

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, values: list[str]) -> list[str]:
        # Quita espacios, descarta vacios y duplicados (sin importar mayusculas),
        # respetando el orden en que aparecieron.
        limpias: list[str] = []
        vistas: set[str] = set()
        for value in values:
            value = value.strip()
            # Ante un texto ambiguo, el modelo tiende a "rellenar" la lista con
            # comodines como "indeterminado" para cumplir la regla de no vacia.
            # Eso no es una tecnologia: se descarta, asi la regla se cumple de verdad.
            if value.lower() in _COMODINES:
                continue
            if value and value.lower() not in vistas:
                vistas.add(value.lower())
                limpias.append(value)
        # Se comprueba DESPUES de limpiar: ["  "] o ["desconocido"] tambien
        # cuentan como lista vacia.
        if not limpias:
            raise ValueError("La lista de tecnologias no puede estar vacia.")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def validar_resumen(cls, value: str) -> str:
        # Se valida aca (y no con min_length en Field) para no agregar al JSON
        # Schema restricciones que algunos proveedores no aceptan.
        value = value.strip()
        if len(value) < 10:
            raise ValueError("El resumen tecnico esta vacio o es demasiado corto.")
        return value


# =============================================================================
# PRE-ENTREGA 3 - Contrato de salida del RAG (ver rag.py).
# =============================================================================
# Texto EXACTO que debe devolver el modelo cuando la respuesta no esta en el
# contexto. Se define una sola vez: lo usan el prompt, el validador y las pruebas.
NO_LO_SE = "No lo sé."


class RespuestaLLM(BaseModel):
    """Lo que el LLM debe devolver. Lo parsea PydanticOutputParser en rag.py.

    Las "description" de cada campo llegan al modelo dentro de las instrucciones
    de formato (parser.get_format_instructions()).
    """

    respuesta: str = Field(
        description=(
            "Respuesta en espanol basada UNICAMENTE en el CONTEXTO. Si la respuesta "
            f'no esta en el contexto, exactamente "{NO_LO_SE}"'
        )
    )
    en_contexto: bool = Field(
        description="true si el CONTEXTO contiene la respuesta; false si no la contiene."
    )
    fuentes: list[str] = Field(
        default_factory=list,
        description=(
            'Ids de los fragmentos del CONTEXTO usados para responder, ej. ["F1", "F3"]. '
            "Lista vacia si en_contexto es false."
        ),
    )

    @field_validator("fuentes")
    @classmethod
    def normalizar_fuentes(cls, values: list[str]) -> list[str]:
        # El modelo a veces escribe "[F1]" o "f1": se normaliza a "F1", sin repetir.
        limpias: list[str] = []
        for value in values:
            value = value.strip().strip("[]").strip().upper()
            if value and value not in limpias:
                limpias.append(value)
        return limpias

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaLLM":
        # FILTRO DE VERACIDAD (segunda linea de defensa, despues del prompt):
        #
        # 1) Si el modelo dice "No lo se" pero marca en_contexto=true, se le cree
        #    al "No lo se": es la opcion segura.
        if self.respuesta.strip().rstrip(".").lower() == NO_LO_SE.rstrip(".").lower():
            self.en_contexto = False
        # 2) Si dice que NO esta en el contexto, cualquier otro texto que haya
        #    escrito seria conocimiento propio (posible alucinacion): se descarta
        #    y se deja exactamente "No lo se.".
        if not self.en_contexto:
            self.respuesta = NO_LO_SE
            self.fuentes = []
            return self
        # 3) Si dice que SI esta en el contexto, tiene que decir de donde lo saco.
        #    Una respuesta sin fuentes no se puede verificar: error -> reintento.
        if not self.fuentes:
            raise ValueError("Una respuesta basada en el contexto debe citar al menos una fuente.")
        if len(self.respuesta.strip()) < 2:
            raise ValueError("La respuesta esta vacia.")
        self.respuesta = self.respuesta.strip()
        return self


class Referencia(BaseModel):
    """Fragmento de un documento que respalda la respuesta."""

    id: str = Field(description="Id del fragmento dentro de la consulta, ej. F1.")
    archivo: str = Field(description="Archivo de data/ del que sale el fragmento.")
    fragmento: int = Field(ge=0, description="Numero de fragmento dentro del archivo (desde 0).")
    distancia: float | None = Field(
        default=None, description="Distancia coseno a la pregunta (menor = mas parecido)."
    )


class RespuestaRAG(BaseModel):
    """Respuesta final de get_rag_response(): el texto y sus referencias."""

    pregunta: str
    respuesta: str
    en_contexto: bool
    referencias: list[Referencia] = Field(default_factory=list)
