import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


"""1.Se debe implementar un clasificador KNN cuando k puede ser cualquier número, después de 
tener mi clasificador utilizamos el ciclo for el cual nos va ayudar a obtener la validación cruzada, 
luego hacemos un split y lo dividimos en 4, luego de que este se entrena con 3 de los grupos y 
vuelve a validar con otro grupo y nos arrojara nuestra matriz de confusión.
n_neighbors= 1 al 18

2.Se implementaron paquetes de las clases pasadas como pandas, numpys y los sklearn

3.Aparte se creo una variable llamada rutina el cual utilizando el comando pd.read_excel para asi 
ubicar el archivo de Excel llamado IA-Feijoas.xlsx

4.Se creo el Scaler para asi utilizar el comando StandardScaler para asi estándarizar los datos 
eliminando la media y escalando los datos de forma que su varianza sea igual a 1.

5. Al crear nuestro ciclo for utilizamos nuestro entreno y nuestra validación para asi al momento 
de que este tome el valor del Excel pueda hacer el entrenamiento y la validación.

6. Se utilizo el comando de confusión_matrix para asi este nos pueda mostrar la matriz de 
confusión. """

ruta=pd.read_excel('E:\ONE DRIVE\OneDrive\Escritorio\Inteligenica artificial\Parcial\IA-Feijoas.xlsx')
print(ruta)


print(ruta.head)



Scaler= StandardScaler()
cm=ruta.columns.values
x=ruta[cm[1:2]].values
y=ruta['bin'].values
print(cm)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=3, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=4, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=5, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=6, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=7, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=9, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=10, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=11, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=12, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=13, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=14, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=15, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

ruta=pd.read_excel('E:\ONE DRIVE\OneDrive\Escritorio\Inteligenica artificial\Parcial\IA-Feijoas.xlsx')
print(ruta)


print(ruta.head)



Scaler= StandardScaler()
cm=ruta.columns.values
x=ruta[cm[1:2]].values
y=ruta['bin'].values
print(cm)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=3, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=4, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=5, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=6, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=7, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=9, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=10, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=11, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=12, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=13, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=14, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=16, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

ruta=pd.read_excel('E:\ONE DRIVE\OneDrive\Escritorio\Inteligenica artificial\Parcial\IA-Feijoas.xlsx')
print(ruta)


print(ruta.head)



Scaler= StandardScaler()
cm=ruta.columns.values
x=ruta[cm[0:2]].values
y=ruta['bin'].values
print(cm)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=1, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=3, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=4, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=5, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=6, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=7, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=9, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=10, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=11, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=12, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=13, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=14, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


SFK=StratifiedKFold(n_splits=4, shuffle=True, random_state=2)
temp=[]
for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=15, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)

for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=16, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)



for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=17, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)


for train,test in SFK.split(x,y):
    entrenamiento = x[train]
    validacion = x[test]
    bin_e=y[train]
    bin_v=y[test]
    Scaler.fit(entrenamiento)
    entrenamiento=Scaler.transform(entrenamiento)
    validacion=Scaler.transform(validacion)
    print('e=',entrenamiento)
    print('v=',validacion)
    
    
    

    KNN=KNeighborsClassifier(n_neighbors=18, p=1)
    KNN.fit(entrenamiento, bin_e)
    bin_p =KNN.predict(validacion)
    print(bin_p, bin_v)
    sc=KNN.score(validacion, bin_v)
    temp.append(sc)
    TP=confusion_matrix(bin_v, bin_p).ravel()
    FN=confusion_matrix(bin_v, bin_p).ravel()
    FP=confusion_matrix(bin_v, bin_p).ravel()
    TN=confusion_matrix(bin_v, bin_p).ravel()
    print( TP, FN, FP, TN)
    fig=plot_confusion_matrix(KNN, validacion,bin_v,display_labels=KNN.classes_)
    plt.show()
MSC= statistics.mean(temp)
print('TASA DE A', MSC)