import numpy as np
import random

def PerceptronBase(features: np.array, labels: np.array, T: int):
    #inicaliza theta com 0
    theta: np.array = np.zeros(features[0].shape)
    theta_0: float = 0

    for _ in range(T): #para cada iteração no dataset
        for _ in range(len(features)): #iterando sob o dataset
            i = random.randint(0, len(features) - 1)

            if (labels[i] * (np.dot(theta, features[i]) + theta_0)) <= 0: #caso aconteça um erro
                theta = theta + (labels[i] * features[i]) #atualizar theta
                theta_0 = theta_0 + labels[i] #atualizar theta_0

    return theta, theta_0

def SGD(features: np.array, labels: np.array, T: int, eta_0: float, lambda_: float):
    #inicaliza theta com 0
    theta: np.array = np.zeros(features[0].shape)
    theta_0: float = 0

    for _ in range(T):
        i = random.randint(0, len(features) - 1)

        eta = eta_0 / (1 + (T / 100))  # ou eta_0 / sqrt(t)

        z = labels[i] * (np.dot(theta, features[i]) + theta_0)

        if (z < 1): #loss é diferente de zero
            theta = theta + (eta * (features[i] * labels[i])) - (eta * (lambda_ * theta))

            theta_0 = theta_0 + (eta * labels[i])
        else: #loss é zero
            theta = theta - eta * (lambda_ * theta)

    return theta, theta_0

def Classificar(theta: np.array, theta_0: float, x: np.array):
    value =  np.dot(theta, x) + theta_0

    if value <= 0:
        return -1
    
    else:
        return 1
    
def Acuracia(theta, theta_0, features, labels):
    acertos: int = 0

    for i in range(len(features)): #para cada exemplo no dataset
        if Classificar(theta, theta_0, features[i]) == labels[i]: #se acertou aumentar o contador de acertos
            acertos += 1

    return acertos / len(features)
     