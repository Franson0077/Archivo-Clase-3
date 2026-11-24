# Manual interno: Trazabilidad de medicamentos

> Documento FICTICIO de ejemplo ("Droguería Ejemplo"), creado para la Pre-entrega 3.
> No describe procedimientos reales de ninguna empresa ni reemplaza a la normativa vigente.

## 1. Qué es la trazabilidad

La trazabilidad permite seguir cada unidad de un medicamento desde el laboratorio que lo
fabricó hasta la farmacia que lo vende. Cada envase trazable tiene un código único que
combina el **GTIN** (código del producto) y un **número de serie**. Con ese código se puede
saber quién tuvo la unidad, cuándo y dónde.

## 2. Responsables

- El **Director Técnico** es responsable de que todos los movimientos de productos
  trazables se informen correctamente.
- El **Analista de Trazabilidad** controla todos los días que no queden eventos pendientes
  o rechazados.
- Cada operario que recibe o despacha mercadería trazable debe escanear todas las unidades;
  no está permitido cargar códigos a mano.

## 3. Eventos que se informan

Cada vez que una unidad trazable cambia de manos o de estado, se informa un "evento" en el
sistema de trazabilidad:

| Situación | Evento | Plazo para informarlo |
|---|---|---|
| Recepción desde un laboratorio | Recepción de producto | Mismo día de la recepción |
| Despacho a una farmacia | Envío a farmacia | Antes de que el camión salga |
| Devolución de una farmacia | Recepción por devolución | Dentro de las 24 horas |
| Producto robado o extraviado | Robo / extravío | Dentro de las 24 horas, con denuncia |
| Producto vencido o destruido | Destrucción | Al recibir el certificado de destrucción |

## 4. Códigos con problemas

- **Código ilegible:** la unidad no se despacha. Se separa y se consulta al laboratorio.
- **Código ya informado por otro agente:** puede indicar una unidad duplicada o falsificada.
  Se separa en cuarentena y se avisa al Director Técnico en el día.
- **Evento rechazado por el sistema:** el Analista de Trazabilidad lo corrige dentro de las
  48 horas. Si no puede corregirlo, lo escala al Director Técnico.

## 5. Auditoría interna

Una vez por mes se elige al azar una muestra de **50 unidades** despachadas y se controla
que todos sus eventos estén informados y coincidan con los remitos. Si el error supera el
2 % de la muestra, se repite el control con 200 unidades y se capacita nuevamente al
personal involucrado.

## 6. Conservación de registros

Los registros de trazabilidad (eventos, remitos escaneados y reportes de auditoría) se
conservan durante **5 años**, en formato digital y con copia de seguridad semanal.
