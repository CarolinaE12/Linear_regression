# Librería NumPy: permite crear vectores de varias dimensiones
import numpy as np

#Librerias y Funciones Sklearn y Matplotlib
from sklearn.model_selection import train_test_split # Se usa para dividir los datos de entrenamiento y prueba
from sklearn.linear_model import LinearRegression # Implementa el modelo de regresión lineal, es lo que permite la predicción de los nuevos datos
import matplotlib.pyplot as plt # Permite crear los elementos gráficos

# Research_Hours Array con las horas de búsqueda
Research_Hours = np.array([5,15,10,20,25,8,30,1218,22,40,28,35,50,45,10,30,20,40,50,16,60,24,36,44,80,56,70,100,90,15,45,30,60,
                           75,24,90,36,54,66,120,84,105,150,135,25,75,50,100,125,40,150,60,90,110,200,140,175,250,225,40,120,80,
                           160,200,64,240,96,144,176,320,224,280,400,360,65,195,130,260,325,104,390,156,234,286,520,364,455,650,
                           585,105,315,210,420,525,168,630,252,378,462,840,588,735,1050,945,950])

# Number_Of_Publications Array con el número de publicaciones
Number_Of_Publications = np.array([3,10,6,12,15,4,18,7,14,16,20,17,19,25,22,6,20,12,24,30,8,36,14,28,32,40,34,38,50,44,9,30,18,36,
                                   45,12,54,21,42,48,60,51,57,75,66,15,50,30,60,75,20,90,35,70,80,100,85,95,125,110,24,80,48,96,120,
                                   32,144,56,112,128,160,136,152,200,176,39,130,78,156,195,52,234,91,182,208,260,221,247,325,286,63,
                                   210,126,252,315,84,378,147,294,336,420,357,399,525,462])


#Transforma el Array Research_Hours en una matriz de una sola columna
x = Research_Hours.reshape(-1,1)

# Asigna la variable y a Number_Of_Publications
y = Number_Of_Publications

# Divide los datos en conjuntos de (80%) entrenamiento y (20%) prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train_test_split divide los datos
# Datos x = Research_Hours, Datos y = Number_Of_Publications
# test_size=0.2 reserva el 20% de los datos como conjunto de prueba, el 80% restante se usa en el entrenamiento
# random_state=42 permite que al ejecutar el modelo se obtenga la misma división


model = LinearRegression()# Model: donde se guarda el modelo, LinearRegression crea una instancia de la clase LinearRegression
model.fit(x_train,y_train)# model.fit método con el cual se entrena el modelo de regresión lineal

y_pred = model.predict(x_test) #Realiza las prediciones sobre los datos de prueba x_test
r2 = model.score(x_test, y_test)# Se calcula el coeficiente de determinación en R2
print("R2 Score:", r2)# Imprime Score R2
coeffient = model.coef_[0] # Guarda el coeficiente de determinación en coeffient
print("Coefficient:", coeffient) # Imprime el coeficiente de determinación
intercept = model.intercept_ # Obtiene el punto de intercepción de los datos
print("Intercept:", intercept) # Imprime el punto de intercepción de los datos

# Gráfica de los datos de prueba al importar la libreria matplotlib.pyplot como plt
plt.scatter(x_test,y_test,color='red')# Crea un diagrama de dispersión con los datos x_test y y_test
plt.plot(x_test,y_pred,color='blue',linestyle = '-',linewidth = 2,label = 'Predicted publications')# Dibuja la línea azul que representa las predicciones del modelo
plt.xlabel('Research Hours')# Añade la etiqueta Research Hours en el eje X
plt.ylabel('Number of Publications')# Añade la etiqueta Number of Publicationss en el eje Y
plt.title('Research Hours vs Number of Publications')#Añade el título Research Hours vs Number of Publications al gráfico
plt.legend() # Crea una etiqueta con un color que ayuda a distinguir la columna
plt.show()# Muestra el gráfico que se genero


#-----------------------------------------------------------------------
# Conclusion del análisis de los datos
# Entre más horas de estudio, mayor número de publicaciones científicas
