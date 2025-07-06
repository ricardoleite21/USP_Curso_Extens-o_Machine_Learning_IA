import numpy as np
from src.lib.feature_map import Polynomial_Regression as PR
from src.lib.feature_map import Combinations_Replacement as CR
from src.lib.feature_map import Length, Min, Max

np.random.seed(42)

#cria a design matrix, sendo ela nossa nova matriz de treino
def Design_Matrix(input_matrix, phi_degree):
    d_y = Length(input_matrix) #range of examples
    d_x = Length(input_matrix[0]) #range of variables

    design_matrix = np.zeros((d_y, CR(d_x, phi_degree) + 1))

    for y in range(d_y): #pra cada exemplo, expande polinomialmente ele
        design_matrix[y] = PR(input_matrix[y], d_x, phi_degree)

    return design_matrix

#calcula a loss pra poder mostrar quanto mais ou menos que tá baixando
def Loss(labels, phi, weights):
    N = Length(phi) 
    loss = 0

    for i in range(N):
        #calculando o dot product primeiro; 
        dot_product = np.dot(weights, phi[i])
        #calculando o desvio do valor esperado;
        error = labels[i] - dot_product
        #calculando o vetor de desvio a ser somado;
        error = pow(error,2)
        loss += error

    return loss/N


#como o objetivo não é trabalhar com uma quantidade gigantesca de
#dados, não é necessário implementar a versão estocástica, bastando
#a vanilla; 
def Gradient_Descent(labels, phi, n1, epochs, theta_size):
    losses = np.zeros((epochs))
    weights = np.random.rand(theta_size)

    for i in range(epochs):
        for t in range(len(phi)):
            weights += n1 * phi[t]*(labels[t]-np.dot(weights, phi[t]))
        losses[i] = Loss(labels, phi, weights)
        
    return weights, losses

