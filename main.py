from fastapi import FastAPI, Query
import joblib

# Cargar el pipeline desde el archivo .pkl
pipeline = joblib.load('pipeline.pkl')

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el endpoint para la predicción
@app.get("/predict/")
async def predict(texto: str = Query(..., description="Texto a clasificar")):
    # Realizar la predicción con el pipeline
    etiqueta = pipeline.predict([texto])
    # Convertir el resultado a int para evitar problemas de serialización
    return {"Etiqueta": int(etiqueta[0])}