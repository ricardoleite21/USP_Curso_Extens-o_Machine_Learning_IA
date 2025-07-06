from LoadDataset import carregar_dataset_bolas
from Model import PerceptronBase, SGD, Acuracia
from visualizacao import visualizar_perceptron
from sklearn.model_selection import train_test_split
import os

# Carrega o dataset
dataset, features, labels = carregar_dataset_bolas(os.path.join("datasets", "bolas.csv"))

# Separação entre dados de treino e dados de teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Treinamento do modelo Perceptron com dados de treino
thetaP, theta_0P = PerceptronBase(X_train, y_train, 20)

# Treinamento do modelo SGD com dados de treino
thetaSGD, theta_0SGD = SGD(X_train, y_train, 200000, 1, 0.5)

# Avaliação dos modelos nos dados de treino
acuraciaP_treino = Acuracia(thetaP, theta_0P, X_train, y_train)
acuraciaSGD_treino = Acuracia(thetaSGD, theta_0SGD, X_train, y_train)

# Avaliação dos modelos nos dados de teste
acuraciaP_teste = Acuracia(thetaP, theta_0P, X_test, y_test)
acuraciaSGD_teste = Acuracia(thetaSGD, theta_0SGD, X_test, y_test)

# Impressão dos resultados
print(f"Perceptron - Acurácia no treino: {acuraciaP_treino:.4f}")
print(f"Perceptron - Acurácia no teste: {acuraciaP_teste:.4f}")
print(f"SGD - Acurácia no treino: {acuraciaSGD_treino:.4f}")
print(f"SGD - Acurácia no teste: {acuraciaSGD_teste:.4f}")

# Visualização dos resultados no conjunto de treino
visualizar_perceptron(X_train, y_train, thetaP, theta_0P, acuraciaP_treino, 
                     salvar_arquivo="perceptron_treino")
visualizar_perceptron(X_train, y_train, thetaSGD, theta_0SGD, acuraciaSGD_treino, 
                     salvar_arquivo="sgd_treino")

# Visualização dos resultados no conjunto de teste
visualizar_perceptron(X_test, y_test, thetaP, theta_0P, acuraciaP_teste, 
                     salvar_arquivo="perceptron_teste")
visualizar_perceptron(X_test, y_test, thetaSGD, theta_0SGD, acuraciaSGD_teste, 
                     salvar_arquivo="sgd_teste")