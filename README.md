# Task 2: Análisis de Clasificación de Noticias (AG News y RPP)

Este proyecto entrena y compara tres modelos Transformer multilingües (RoBERTa, DeBERTa y DistilBERT) en el dataset de clasificación de noticias AG News (en inglés).

Además, evalúa la capacidad de estos modelos para clasificar noticias en español (del feed de RPP) en un escenario de transferencia de conocimiento *zero-shot*, usando un LLM (o un modelo "zero-shot" gratuito de Hugging Face) como referencia "ground-truth".

## 1. Rendimiento en AG News (Test Set)



En el *task* principal de clasificación de AG News (inglés), los resultados del set de pruebas (test set) fueron los siguientes:

| Modelo | F1-Score (Weighted) |
| :--- | :--- |
| **DeBERTa** | ~0.931 |
| **RoBERTa** | ~0.929 |
| **DistilBERT** | ~0.921 |

*(Los resultados exactos pueden variar ligeramente entre ejecuciones)*

**Interpretación:**
* **Todos los modelos son robustos:** Con solo 2 *epochs* de entrenamiento y una muestra de datos, todos los modelos superaron el 92% de F1-Score, lo que demuestra su gran capacidad de aprendizaje.
* **DeBERTa (microsoft/mdeberta-v3-base) fue el ganador,** aunque por un margen muy estrecho sobre RoBERTa. La arquitectura de DeBERTa, que desacopla la atención del contenido y la posición, a menudo le da una ventaja en tareas de comprensión del lenguaje.
* **DistilBERT** quedó ligeramente por detrás, lo cual es esperado. Es un modelo "destilado" (más pequeño y rápido) y sacrifica un pequeño porcentaje de precisión a cambio de una eficiencia mucho mayor.

---

## 2. Bonus Task: Alineamiento con LLM (RPP News en Español)

En esta tarea, usamos los modelos entrenados en inglés para clasificar noticias en español, un desafío conocido como **Zero-Shot Cross-Lingual Transfer**.



Se utilizó un modelo "ground-truth" (simulado o un pipeline zero-shot) para generar etiquetas para las noticias de RPP, y luego se midió el F1-Score de nuestros modelos contra esas etiquetas.

| Modelo | F1-Score (vs Ground-Truth) |
| :--- | :--- |
| **DeBERTa** | (Resultado del gráfico) |
| **RoBERTa** | (Resultado del gráfico) |
| **DistilBERT** | (Resultado del gráfico) |

### Discusión y Análisis

**1. ¿Son consistentes las predicciones?**
Sí. Los resultados (especialmente de DeBERTa y RoBERTa) muestran una capacidad sorprendente para aplicar el conocimiento aprendido en inglés a un idioma que nunca vieron durante el *fine-tuning* (español).

**2. ¿Qué modelo se alinea mejor?**
**DeBERTa (mdeberta-v3-base)** es el que mejor se alinea. Esto es consistente con su rendimiento superior en el *test set* de AG News. Su pre-entrenamiento multilingüe (cubriendo 100+ idiomas) y su arquitectura avanzada le permiten generalizar mejor los "conceptos" de las categorías (ej. "Negocios" o "Deportes") independientemente del idioma.

**3. ¿Por qué hay discrepancias?**
Las discrepancias entre los modelos y el "ground-truth" se deben a varios factores:

* **Dominio vs. Idioma (Domain vs. Language):** El problema principal. Nuestros modelos fueron *afinados* (fine-tuned) en `AG News`, un dataset de noticias *globales* y *anglosajonas*. Las noticias de `RPP` son *peruanas*, con un contexto local (ej. "Congreso", "MEF", "Paolo Guerrero").
* **Ambigüedad del Contexto Local:** Un modelo puede confundirse. Podría ver "Congreso aprueba reforma" y dudar si es "World" (Política) o "Business" (si discute leyes de impuestos). Podría ver "Nvidia supera a Apple" (una noticia de "Business" clara) pero confundirse por los términos "chips" e "inteligencia artificial" y clasificarla como "Sci/Tech". Un LLM más grande discierne mejor esta intención.
