# Prediccion de Precios de Inmuebles en Florida: Un Enfoque Multimodal

Este repositorio contiene un pipeline completo de Machine Learning de punta a punta desarrollado para una competencia de prediccion de precios en el mercado inmobiliario de Florida (dataset de Zillow). El proyecto implementa una estrategia multimodal, aprovechando datos tabulares, imagenes de las propiedades y las descripciones de los listados para predecir los valores de venta utilizando validacion Out-of-Fold (OOF).

La arquitectura final consta de tres modelos independientes combinados a traves de un ensamble de mezcla ponderada optimizado.

Estructura del Repositorio
- 01_modelo_tabular_lgbm_optuna.ipynb: Procesamiento de datos tabulares, ingenieria de variables con Optuna y entrenamiento de gradiente aumentado.
- 02_modelo_imagenes_embeddings_clip.ipynb: Extraccion de caracteristicas visuales con CLIP, reduccion de dimensiones mediante PCA y evaluacion del modelo de imagenes.
- 03_modelo_texto_tfidf_svd.ipynb: Procesamiento del lenguaje natural de los anuncios de Zillow mediante tecnicas de mineria de texto y modelo base.
- 04_ensemble_multimodal_blending.ipynb: Formulacion del ensamble final a traves de la busqueda de pesos optimos en grilla tridimensional.

## Descripcion General de la Arquitectura

                      ┌───────────────────┐
                      │    Datos Zillow   │
                      └─────────┬─────────┘
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
 │   Tabular   │         │  Imagenes   │         │    Texto    │
 └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
        │                       │                       │
        ▼                       ▼                       ▼
 ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
 │ LightGBM    │         │ CLIP (ViT)  │         │ TF-IDF +    │
 │ CatBoost    │         │    + PCA    │         │    SVD      │
 │ XGBoost     │         └──────┬──────┘         └──────┬──────┘
 └──────┬──────┘                │                       │
        │                       ▼                       ▼
        │                ┌─────────────┐         ┌─────────────┐
        │                │  LightGBM   │         │  LightGBM   │
        ▼                └──────┬──────┘         └──────┬──────┘
 ┌─────────────┐                │                       │
 │ Preds OOF   │◄───────────────┴───────────────────────┘
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Blending   │
 │   Grid3D    │
 └──────┬──────┘
        ▼
┌───────────────────────────────┐
│ Prediccion Multimodal Final   │
└───────────────────────────────┘

---

## Descripcion de los Modelos Principales

### 1. Modelo Tabular (LightGBM + CatBoost + XGBoost)
* Ingenieria de Caracteristicas: Procesamiento espacio-temporal avanzado que incluye Geo-clustering (KMeans con 30 clusters) para capturar micro-mercados, Target Encoding estricto mediante K-Fold para evitar la fuga de datos (data leakage) y ratios financieros calculados por codigo postal. Los valores atipicos se manejaron mediante la winsorizacion de la variable objetivo (percentiles 1-99).
* Optimizacion: La sintonizacion de hiperparametros se realizo con Optuna a lo largo de 40/80 pruebas, optimizando directamente la métrica de error porcentual absoluto medio (MAPE) con parada temprana (early stopping).
* Resultado: Se alcanzo un MAPE OOF independiente de aproximadamente 25.35%.

### 2. Modelo de Vision por Computadora (OpenCLIP + LightGBM)
* Extraccion de Caracteristicas: Procesamiento y extraccion de semantica visual de alta dimension utilizando OpenCLIP (ViT-B-32 entrenado por OpenAI). Los embeddings de multiples fotografias por propiedad se promediaron matematicamente para capturar el contexto completo del inmueble.
* Alineacion Semantica: Se diseño una caracteristica personalizada de alineacion texto-visual tokenizando descriptores de la propiedad (comodidades, estado) y calculando la similitud coseno entre el tensor de la imagen y la representacion del texto.
* Reduccion de Dimensiones: Compresion del vector combinado visual/semantico de 1025 dimensiones a 128 componentes densos mediante PCA truncado, ajustado estrictamente dentro de cada fold para asegurar la honestidad de la validacion.

### 3. Modelo de Procesamiento de Lenguaje Natural (TF-IDF + Truncated SVD)
* Mineria de Texto: Procesamiento de las descripciones de texto extraidas de los anuncios inmobiliarios utilizando TfidfVectorizer (maximo de 500 caracteristicas, unigramas y bigramas, excluyendo palabras de parada en ingles).
* Analisis Semantico Latente: Reduccion de la dimensionalidad a 50 componentes densos mediante Descomposicion en Valores Singulares (SVD) antes de alimentar las caracteristicas en un regresor LightGBM optimizado.

---

## Estrategia de Blending y Ensamble

Para aprovechar los patrones unicos capturados por cada formato estructural de datos, se implemento una busqueda en grilla 3D (Grid Search Blending) optimizada sobre las predicciones Out-of-Fold (OOF).

El ensamble final sincronizo con exito las metricas economicas tabulares con las caracteristicas visuales y textuales no estructuradas, proporcionando una puntuacion de validacion altamente robusta de cara al conjunto de prueba de la competencia.


```python
# Pesos optimos encontrados por la busqueda en grilla:
W_TABULAR = 0.98
W_IMAGES  = 0.01
W_TEXT    = 0.01

final_blend = (W_TABULAR * oof_lgbm) + (W_IMAGES * oof_cb) + (W_TEXT * oof_xgb)


