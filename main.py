from fastapi import FastAPI, Query
import pickle
from tensorflow.keras.models import load_model
import re
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el tokenizador
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizador cargado desde 'tokenizer.pickle'")

# Cargar el modelo entrenado
model_path = 'saved_models/lstm_model.h5'
model = load_model(model_path)
print(f"Modelo cargado desde {model_path}")

# Inicializar spaCy en español
nlp = spacy.load("es_core_news_sm")

# Diccionario de emojis específicos y sus significados en español
emoji_dict = {
    '😂': 'risa', '👏': 'aplauso', '❤': 'amor', '✨': 'brillo', '🤣': 'carcajada',
    '🙌': 'apoyo', '🌟': 'estrella', '😢': 'tristeza', '💫': 'resplandor', '🔥': 'fuego',
    '😍': 'amor', '🤮': 'asco', '💪': 'fuerza', '🙄': 'recelo', '🙏': 'rezar',
    '😮': 'sorpresa', '😭': 'llanto', '🎨': 'arte', '🌈': 'arcoiris', '😡': 'enfado',
    '📸': 'fotografía', '💩': 'mierda', '🎯': 'objetivo', '🚀': 'impulso', '🤷': 'indiferencia',
    '🤡': 'payaso', '⭐': 'estrella', '😘': 'beso', '🏳️': 'bandera', '🤦': 'frustración',
    '✊': 'puño', '😔': 'decepción', '👌': 'ok', '💯': 'perfecto', '💜': 'corazón morado',
    '👍': 'bien', '🌱': 'crecimiento', '😎': 'genial', '💀': 'calavera', '😅': 'alivio',
    '🤔': 'pensando', '🤑': 'dinero', '💕': 'amor', '💝': 'amor'
}

# Función de preprocesamiento
def preprocess_text(text):
    text = text.lower()
    for emoji_char, emoji_desc in emoji_dict.items():
        text = text.replace(emoji_char, emoji_desc)
    text = re.sub(r":(\w+):", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    
    additional_stopwords = {"el", "la", 'los', 'las', 'unos', 'unas', "un", "una", "lo", "que"}
    stopwords = STOP_WORDS.union(additional_stopwords)
    
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stopwords and not token.is_punct and not token.is_space and len(token.lemma_) > 3
    ]
    return " ".join(tokens)

def preprocess_input(text, tokenizer, max_length):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el endpoint para la predicción
@app.get("/predict/")
async def predict(texto: str = Query(..., description="Texto a clasificar")):
    processed_input = preprocess_input(texto, tokenizer, max_length=100)  # Ajusta max_length según tu modelo
    prediction = model.predict(processed_input)
    prediction_class = (prediction > 0.5).astype(int)
    return {"Etiqueta": int(prediction_class[0][0])}
