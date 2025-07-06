import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

# rede neural com 2 layers (só 1 é hidden)
class NeuralNet:
    def __init__(self, learning_rate, dataset, labels, epochs):
        self.lr = learning_rate
        self.ds = dataset
        self.labels = labels
        self.epochs = epochs
        self.cost = np.zeros(epochs)

        # weights e biases initalization 
        self.W1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10,1) - 0.5

    # tipo de Loss: Cross-Entropy multiclasse / SoftMax Loss
    def loss(self, truth, A2):
        m = truth.shape[1]
        return -np.sum(np.multiply(truth,np.log(A2))) / m

    # função de ativação: ReLU
    def relu(self, x):
        return np.maximum(x, 0)
    
    # SoftMAX: traduz tudo para probabilidades entre as classes
    def softmax(self, Z):
        exp = np.exp(Z)
        return exp / sum(exp)
    
    # atualiza viés e pesos com base no gradient descent
    def update(self, nb1, nb2, nW1, nW2):
        self.W1 = self.W1 - (self.lr * nW1)
        self.W2 = self.W2 - (self.lr * nW2)
        self.b1 = self.b1 - (self.lr * nb1)
        self.b2 = self.b2 - (self.lr * nb2)

    # Feed-Forward: como processar os inputs
    def forward_prop(self, input = None):
        if input is None:
            input = self.ds
        # 1° layer (inputs)
        Z1 = (self.W1).dot(input) + self.b1
        A1 = self.relu(Z1)
        # 2° layer (hidden)
        Z2 = (self.W2).dot(A1) + self.b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    # derivada da função ReLU
    def drelu(self, Z):
        return (Z > 0).astype(float)
    
    # cria matriz de (labels)X(exemplos), com '1' na label 
    # correspondente do exemplo
    def one_hot(self, Y):
        hot = np.zeros((Y.size, Y.max() + 1))
        hot[np.arange(Y.size), Y] = 1

        return hot.T

    # Back-Propagation: cálculo do gradiente na rede neural
    def back_prop(self, Z1, A1, Z2, A2):
        m = self.labels.size
        hotY = self.one_hot(self.labels)
        
        dZ2 = A2 - hotY
        dW2 = dZ2.dot(A1.T) / m
        db2 = np.sum(dZ2) / m
        
        dZ1 = (self.W2.T).dot(dZ2) * self.drelu(Z1)
        dW1 = dZ1.dot(self.ds.T) / m
        db1 = np.sum(dZ1) / m
        
        # retorna os nudges
        return dW1, db1, dW2, db2
        
    # treinando nosso modelo com o GOAT    
    def gradient_descent(self):
        # processa todo o dataset a cada loop
        for i in range(self.epochs):
            Z1, A1, Z2, A2 = self.forward_prop()
            hotY = self.one_hot(self.labels)
            custo = self.loss(hotY, A2)
            self.cost[i] = custo
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2)
            self.update(db1, db2, dW1, dW2) 

            if (i % 10 == 0):
                print("Epoch: ", i, " Loss: ", custo)

    # para um singelo número de input, retorna classe prevista
    def predict(self, image):
        _, _, _, A2 = self.forward_prop(image)
        

        return (np.argmax(A2), A2)

    # escolha um index aleatório, e veja se a NN acerta!
    def rand_predict(self, index):
        current = self.ds[:, index].reshape(-1, 1)
        prediction, probs = self.predict(current)
        label = self.labels[index]

        print("Acho que é: ", prediction)
        print("Label do desenho: ", label)
        
        # plotando número
        current = current.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current, interpolation='nearest')
        plt.show()

        # gráfico das probabilidades de cada classe
        self.plot_vector_as_bars(probs.squeeze())

    # usando o dataset de testes pra ver quão bem o bichão tá funcionando
    def accuracy(self, Y_test, X_test):
        total = X_test.shape[1]
        
        # propagando os testes pra ver o que sai
        _, _, _, A2 = self.forward_prop(X_test)
        predictions = np.argmax(A2, axis=0)
        
        # comparando com labels e vendo o quanto acertou
        correct = np.sum(predictions == Y_test)
        accuracy = (correct / total) * 100
        error_rate = 100 - accuracy
        
        print(f"Acurácia: {accuracy:.2f}%")
        print(f"Taxa de Erro: {error_rate:.2f}%")

    # funções meio inúteis abaixo (só plotagem)

    def plot_vector_as_bars(self, vector):
        plt.figure(figsize=(8, 4))
        bars = plt.bar(range(10), vector, color='skyblue', edgecolor='black')
        
        max_idx = vector.argmax() if hasattr(vector, 'argmax') else np.argmax(vector)
        bars[max_idx].set_color('salmon')
        
        plt.xticks(range(10))
        plt.xlabel('Classes')
        plt.ylabel('Probabilidade')
        plt.title('Distribuição')
        plt.ylim(0, 1.1 if max(vector) <= 1 else None)  
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def showLoss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost, 'o-', color='blue', linewidth=2, markersize=5)
        
        plt.title("Evolução da função Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.show()