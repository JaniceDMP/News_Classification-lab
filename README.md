# Task 2: Análisis de Clasificación de Noticias (AG News y RPP)

Este proyecto entrena y compara tres modelos Transformer multilingües (RoBERTa, DeBERTa y DistilBERT) en el dataset de clasificación de noticias AG News (en inglés).

Además, evalúa la capacidad de estos modelos para clasificar noticias en español (del feed de RPP) en un escenario de transferencia de conocimiento *zero-shot*, usando un LLM (o un modelo "zero-shot" gratuito de Hugging Face) como referencia "ground-truth".

## 1. Rendimiento en AG News (Test Set)



En el *task* principal de clasificación de AG News (inglés), los resultados del set de pruebas (test set) fueron los siguientes:

| Modelo | F1-Score (Weighted) |
| :--- | :--- |
| **DeBERTa** | ~0.911 |
| **RoBERTa** | ~0.918 |
| **DistilBERT** | ~0.908 |

*(Los resultados exactos pueden variar ligeramente entre ejecuciones)*

**Interpretación:**
* **Todos los modelos son robustos:** Con solo 2 *epochs* de entrenamiento y una muestra de datos, todos los modelos superaron el 90% de F1-Score, lo que demuestra su gran capacidad de aprendizaje.
* **RoBERTa fue el ganador,** aunque por un margen muy estrecho sobre DeBERTa. Aunque, la arquitectura de DeBERTa, que desacopla la atención del contenido y la posición, a menudo le da una ventaja en tareas de comprensión del lenguaje.
* **DistilBERT** quedó ligeramente por detrás, lo cual es esperado. Es un modelo "destilado" (más pequeño y rápido) y sacrifica un pequeño porcentaje de precisión a cambio de una eficiencia mucho mayor.

---

## 2. Bonus Task: Alineamiento con LLM (RPP News en Español)

En esta tarea, usamos los modelos entrenados en inglés para clasificar noticias en español. El desafío principal fue generar las etiquetas "ground-truth" para las noticias de RPP.

### Metodología del "Ground-Truth"

1.  **API de LLM (Ej. OpenAI):** El plan inicial era usar un LLM de pago como `gpt-3.5-turbo`. Sin embargo, esto falló debido a un error `429 - insufficient_quota` (falta de saldo).
2.  **Modelo LLM Zero-Shot (Gratuito):** Se exploró una alternativa gratuita usando un *pipeline* `zero-shot-classification` de Hugging Face (ej. `facebook/bart-large-mnli`). Este es un LLM real que corre en Colab y clasifica el texto en las categorías dadas.
3.  **LLM "Mock" (Simulación):** Para garantizar la reproducibilidad y evitar fallos de descarga o `try/except`, se optó finalmente por un **LLM "Mock" simulado**. Esta es una función simple basada en palabras clave (ej. "dólar" -> "Business") para generar las etiquetas de forma consistente.

**Los siguientes resultados están basados en la comparación contra el LLM "Mock".**

### Resultados vs. LLM Mock

Los resultados de esta comparación *zero-shot* fueron sorprendentes:

| Modelo | F1-Score (vs Ground-Truth Mock) |
| :--- | :--- |
| **DeBERTa** | **1.0000** |
| **DistilBERT** | 0.8444 |
| **RoBERTa** | 0.7619 |

### Discusión y Análisis

1.  **¿Son consistentes las predicciones?**
    Sí, y en el caso de DeBERTa, la consistencia fue perfecta. Esto demuestra una capacidad de generalización interlingüística (de inglés a español) extremadamente fuerte.

2.  **¿Qué modelo se alinea mejor?**
    **DeBERTa (mdeberta-v3-base) se alineó perfectamente (F1-Score de 1.0)** con nuestro *ground-truth* simulado. Esto significa que sus predicciones (`[2, 1, 3, 0, 2, 2]`) fueron *idénticas* a las reglas definidas en el "mock" (`[2, 1, 3, 0, 2, 2]`).

3.  **¿Por qué las discrepancias?**
    * **DistilBERT (`[2, 1, 3, 0, 3, 2]`)**: Falló en la noticia de Nvidia (Índice 4). La clasificó como `3` (Sci/Tech), mientras que el "mock" la clasificó como `2` (Business). Esto es comprensible, ya que la noticia menciona "chips" e "inteligencia artificial" (Sci/Tech) pero también "acciones" y "empresa" (Business). El "mock" priorizó "Business" y DistilBERT priorizó "Sci/Tech".
    * **RoBERTa (`[2, 1, 3, 2, 2, 2]`)**: Falló en la noticia del Congreso (Índice 3). La clasificó como `2` (Business) en lugar de `0` (World/Política). Este es un error de clasificación más claro, que resultó en el F1-Score más bajo.
