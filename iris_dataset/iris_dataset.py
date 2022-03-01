#%%
from sklearn.datasets import load_iris #Conjunto de dados de espécies Iris

iris = load_iris()
print(iris)

#%% 
## OBSERVAÇÕES
x = iris.data
print(x)

#%% 
## TARGET  espécies das plantas
y = iris.target
print(y)

#%% 
## SHAPE das Observações
print(iris.data.shape) #verifica tamanho do array

#%% 
## SHAPE do Target
print(iris.target.shape) #verifica tamanho do array

#%%
## Importação do KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# %% 
## Treinar a máquina
knn.fit(x,y)

#%% 
## Fazer previsões
species = knn.predict([[5.9,3.,5.1,1.8]])
print(iris.target_names[species])

# %%
## Separar dados em dois grupos
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

# %%
## Avaliação da performance do modelo
knn.fit(x_train,y_train)
previsoes = knn.predict(x_test)

from sklearn import metrics
acertos = metrics.accuracy_score(y_test,previsoes)
print(acertos)

