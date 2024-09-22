#Librerias
import pandas as pd # permite la manipuacion y analisis de los datos
from sklearn.model_selection import train_test_split #permite dividir el conjunto  de datos en subconjuntos aleatorios de entrenamiento y prueba.
from sklearn.linear_model import LogisticRegression #Modelo de aprendizaje automático que se usa en clasificación binaria
from sklearn.metrics import accuracy_score#Permite calcular la precisión del modelo de clasificación

# Cargar los datos
#Usa la biblioteca pandas para poder leer el archivo cvs y
# almacenar el contenido en un DataFrame (data).
data = pd.read_csv('crop_production.csv')

X = data[['FertilizerUsed', 'Rainfall']]  #Crea un nuevo DataFrame llamado x
data['HighYield'] = (data['CropYield'] > 3000).astype(int) # Crea una columna llamada HighYield. 
#Se calcula el valor de cropYield y si el valor es mayor a 3000 devuelve true en caso de que sea menor devuelve falce.
y = data['HighYield']# se crea una variable y se le asigna los valores de HighYield


# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Matriz de confusión y evaluación
conf_matrix = confusion_matrix(y_test, y_pred)#calcula la matriz de confusión y la amacena en conf_matrix
plt.figure(figsize=(8,6))# crea el comtenedor de los elementos graficos
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=True)#crea un mapa de calor usando la bibliteca Seaborn  lo cual permite visualizar la matriz de confusión
plt.xlabel('Predicted')#crea la etiqueta en el eje x
plt.ylabel('Actual')#crea la etiqueta en el eje y
plt.title('Confusion Matrix')#Crea el titulo de la grafica
plt.show()#Muestra las figuras creadas anteriormente

# Imprimir el informe de clasificación y la exactitud
print(classification_report(y_test, y_pred))#genera un informe 
accuracy = accuracy_score(y_test, y_pred)#Calcula la precisión del modelo
print(f'Exactitud del modelo: {accuracy*100:.2f}%')#Imprime la exactitud del modelo