import numpy as np
from sklearn.datasets import load_iris #Conjunto de dados de espécies Iris

iris = load_iris()

## OBSERVAÇÕES
x = iris.data
 
## TARGET  espécies das plantas
y = iris.target

## SHAPE das Observações
#print(iris.data.shape) #verifica tamanho do array
 
## SHAPE do Target
#print(iris.target.shape) #verifica tamanho do array

## Separar dados em dois grupos
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

aux = np.zeros(50)
for k in range(0,25):
    k_ = k+1
    ## Importação do KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k_)

    ## Avaliação da performance do modelo
    knn.fit(x_train,y_train)
    previsoes = knn.predict(x_test)

    from sklearn import metrics
    acertos = metrics.accuracy_score(y_test,previsoes)
    aux[2*k] = k_
    aux[2*k+1] = acertos
    
print(aux)

