import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Descomente a linha abaixo na primeira execução para baixar as stopwords
#nltk.download('stopwords')

def preprocess_text(text):
    """
    Função para pré-processar texto:
    - Converte para minúsculas
    - Remove caracteres especiais e números
    - Remove stopwords
    """
    if isinstance(text, str):
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remover stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return " ".join(words)
    else:
        return ""

def create_bow_vectorizer(conjunto_textos):
    """
    Função para criar o vetorizador Bag of Words a partir do CSV
    """
    # Pré-processar os textos
    conjunto_textos_processados = conjunto_textos.apply(preprocess_text)
    
    # Criar o Bag of Words usando CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(conjunto_textos_processados)
    
    return vectorizer

def text_to_bow_vector(text, vectorizer):
    """
    Função para converter um texto em vetor Bag of Words
    """
    # Pré-processar o texto
    processed_text = preprocess_text(text)
    
    # Transformar em vetor
    vector = vectorizer.transform([processed_text])
    
    # Converter para array ou DataFrame
    return vector.toarray()[0]

def GetFeatures(texts, vectorizer):
    textos_preprocessados = texts.apply(preprocess_text)
    matriz_completa = vectorizer.transform(textos_preprocessados)

    # Converter para array denso se necessário
    features = matriz_completa.toarray()

    return features