1.Se debe implementar un clasificador KNN cuando k puede ser cualquier número, después de 
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
confusión
