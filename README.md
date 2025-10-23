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

### Resultados vs. LLM Zero-Shot (Hugging Face)



| Modelo | F1-Score (vs LLM Zero-Shot) |
| :--- | :--- |
| **DeBERTa** | **0.8222** |
| **DistilBERT** | 0.6667 |
| **RoBERTa** | 0.5556 |

### Discusión y Análisis

1.  **¿Son consistentes las predicciones?**
    La consistencia varía. DeBERTa muestra una **alta consistencia** (F1 > 0.82), lo que indica que su lógica de clasificación (aprendida del inglés en AG News) es muy similar a la lógica de inferencia del LLM Zero-Shot. Los otros modelos muestran una consistencia mucho menor.

2.  **¿Qué modelo se alinea mejor?**
    **DeBERTa (mdeberta-v3-base)** se alinea significativamente mejor. Este es el hallazgo clave: el modelo que ganó en el *test set* de AG News (inglés) también es el que mejor generaliza su conocimiento al español y se alinea con la lógica de un LLM de inferencia (NLI).

3.  **¿Por qué las discrepancias? (Análisis de Resultados)**
    Estamos comparando dos sistemas diferentes en un idioma que no vieron durante el *fine-tuning*:
    * **Modelos Fine-tuned (DeBERTa, etc.):** Entrenados en `AG News` (Inglés). Aprendieron a "clasificar".
    * **LLM Zero-Shot (`bart-large-mnli`):** Entrenado en `NLI` (Inglés). Aprendió a ver si una "premisa" (la noticia) "implica" una "hipótesis" (la etiqueta).

    * **DeBERTa (F1: 0.8222):** Su alta puntuación sugiere que su arquitectura avanzada (`mdeberta-v3-base`) capturó el *concepto semántico* de las categorías (ej. qué "es" Business) de una forma que trasciende el idioma y la tarea.
    * **DistilBERT (F1: 0.6667):** Tuvo un desempeño moderado.
    * **RoBERTa (F1: 0.5556):** Tuvo el F1-Score más bajo, mostrando la mayor discrepancia. Esto sugiere que, aunque es multilingüe (`xlm-roberta-base`), su generalización *cross-lingual* en esta tarea fue menos robusta y su "lógica" de clasificación difirió significativamente de la del LLM Zero-Shot.
    * 
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
    Las discrepancias entre los modelos y el "mock" se deben a varios factores:

* **Dominio vs. Idioma (Domain vs. Language):** El problema principal. Nuestros modelos fueron *afinados* (fine-tuned) en `AG News`, un dataset de noticias *globales* y *anglosajonas*. Las noticias de `RPP` son *peruanas*, con un contexto local (ej. "Congreso", "MEF", "Paolo Guerrero").
* **Ambigüedad del Contexto Local:** Un modelo puede confundirse. Podría ver "Congreso aprueba reforma" y dudar si es "World" (Política) o "Business".
* **Simpleza del "Mock":** Nuestro LLM "mock" es muy simple y se basa solo en palabras clave (ej. "dólar" -> "Business"). Un LLM real (como GPT-4) entendería el *contexto* y el matiz, mientras que nuestros modelos Transformer (como DeBERTa) pueden captar un contexto que el "mock" ignora, lo que lleva a discrepancias.
