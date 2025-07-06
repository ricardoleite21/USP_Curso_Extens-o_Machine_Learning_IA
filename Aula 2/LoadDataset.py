import pandas as pd

def carregar_dataset_bolas(caminho_arquivo):
    """
    Carrega o arquivo CSV e separa em dois arrays numpy:
    - Um array 2D com as features
    - Um array 1D com as classes (+1 ou -1)
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo CSV
        
    Returns:
        tuple: (X, y) onde X é o array de features e y é o array de classes
    """
    # Carregando o dataset usando pandas
    df = pd.read_csv(caminho_arquivo)
    
    # Extraindo as features (feature1 e feature2) para um array numpy 2D
    features = df[['feature1', 'feature2']].to_numpy()
    
    # Extraindo a coluna 'class' para um array numpy 1D
    labels = df['class'].to_numpy()
    
    return df, features, labels

def carregar_dataset_sentimento(caminho_arquivo):
    # Carrega o CSV com as colunas específicas
    df = pd.read_csv(caminho_arquivo, encoding='utf-8')
    
    #pega a coluna de texto
    texts = df["text"]

    # Extraindo a coluna 'sentiment' para um array numpy 1D
    labels = df["sentiment"].to_numpy()

    return texts, labels