import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ANN import NeuralNet as NN
import pandas as pd
import os

# imagens: formato PNG, com dimensões 28 x 28 -> 784 bytes
# entrada: 784 neurônios;
# saída: 10 neurônios (10 classes);

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, '..', 'dataset', 'train.csv')

# carregando dataset + labels
data = pd.read_csv(dataset_path)

# dataset: (70 000, 784)
dataset = data.iloc[:, 1:].values
dataset = dataset.T / 255 # normalizando (MAX_pixel = 255)

# labels 
labels = data.iloc[:, 0].values

# misturando todas as imagens (shuffle geral)
indices = np.arange(dataset.shape[1])
np.random.shuffle(indices)
dataset = dataset[:, indices]
labels = labels[indices]

# dados de treino (6/7) e de validação (1/7)
test_X = dataset[:, :10000]
test_Y = labels[:10000]
train_X = dataset[:, 10000:]
train_Y = labels[10000:]

# hiperparâmetros básicos
lr = float(input("Learning rate: ")) # algo entre 0.5 a 0.05
epochs = int(input("Quanto treinar: ")) # não mais que 600 (demora muito)

# inicializando e treinando a rede neural
network = NN(lr, train_X, train_Y, epochs)
network.gradient_descent()

# checando acurácia total do modelo e exibindo Loss
network.showLoss()
network.accuracy(test_Y, test_X)

# interações engraçadinhas com a rede neural =D
guesses = int(input("Quantas vezes você quer fazer perguntas para mim? "))
for i in range(guesses):
    guess = int(input("Escolha um índice: "))
    network.rand_predict(guess)