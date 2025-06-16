# Predicción de Precios de Viviendas

Este proyecto demuestra un flujo de trabajo sencillo de ingeniería de machine learning para predecir precios de viviendas usando el conjunto de datos *California Housing*. El código está basado en el notebook ubicado en `notebooks/house_price_prediction.ipynb`, y ha sido convertido en scripts reutilizables.

## Estructura del Proyecto

```
├── data
│   ├── raw            # Conjunto de datos original descargado
│   ├── processed      # División de Train/Test
│   └── scores         # Métricas de evaluación
├── models             # Modelos entrenados
├── notebooks          # Jupyter notebooks
├── src                # Código fuente del pipeline
├── requirements.txt   # Dependencias de Python
└── setup.py           # Hace que el proyecto sea instalable
```

## Configuración

1. Crea un entorno virtual e instala las dependencias:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Instala el paquete en modo editable:

```bash
pip install -e .
```

## Uso

El proceso completo puede ejecutarse con los siguientes comandos:

```bash
python src/make_dataset.py       # Descarga y preparación de datos
python src/train.py              # Entrena el modelo
python src/evaluate.py           # Evalúa el modelo
```

Para hacer predicciones sobre nuevos datos:

```bash
python src/predict.py --input ruta/a/datos.csv --output predicciones.csv
```

## Automatización

Se proporciona un `Makefile` para automatizar el flujo de trabajo:

```bash
make all     # ejecuta preparación de datos, entrenamiento y evaluación
```

#Usuario

Jeanpier Miguel Garay Pastrana
Victor Alfonso Ochoa Flores
Shalom Anderson López Melgar



