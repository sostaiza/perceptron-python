# perceptron-python
En este artículo, veremos cómo entrenar un Perceptrón (la unidad básica de una red neuronal) para que aprenda a comportarse como estas compuertas. 

1. Introducción
Las compuertas lógicas AND y OR son bloques fundamentales en la computación y la electrónica digital. En este artículo, veremos cómo entrenar un Perceptrón (la unidad básica de una red neuronal) para que aprenda a comportarse como estas compuertas.

Usaremos Python y algunas librerías como NumPy para manejar la lógica de entrenamiento, Matplotlib para visualizar el error, y un poco de lógica propia para detener el entrenamiento de forma temprana (early stopping). Además, hemos implementado una función que permite al usuario ingresar valores en tiempo de ejecución para probar la red.

2. ¿Qué es un Perceptrón?
El Perceptrón es el modelo de neurona artificial más sencillo. Consiste en:

Pesos (weights) que multiplican las entradas.
Un sesgo (bias) que se suma al resultado.
Una función de activación, en este caso la sigmoid (o logística).

y=activation(w⋅x+b).

Con la activación sigmoide, definimos:

Si la salida es mayor o igual a 0.5, la consideraremos como un 1; de lo contrario, será 0.

3. Explicación Paso a Paso
Imports y dataclasses:
numpy para operaciones matemáticas,
matplotlib.pyplot para graficar,
dataclasses.dataclass para definir una clase inmutable Perceptron (congelada con frozen=True).

Clase Perceptron
Contiene los atributos weights y bias.
El método estático activation implementa la función sigmoide.
El método predict devuelve 0 o 1 según la salida de la sigmoide (con un umbral de 0.5).

train_neuron
Recibe el tipo de compuerta (AND o OR) y parámetros para controlar el entrenamiento (tasa de aprendizaje, épocas máximas, etc.).
Define los datos de entrada/salida para la compuerta elegida.
Inicializa pesos y bias aleatoriamente.
En cada época, calcula la salida, el error y actualiza los pesos y el bias mediante gradiente descendente.
Guarda el historial de error para luego graficarlo.
Implementa una lógica de early stopping en base a:
Cantidad de épocas sin mejora.
Error por debajo de un umbral.
Variación mínima con respecto a la media de un número fijo (avg_window) de épocas anteriores.

visualize_training
Grafica la evolución del error por época.

get_user_input
Solicita al usuario un par de valores (0/1) para probar la red neuronal una vez entrenada.

main
Menú interactivo para elegir la compuerta a entrenar o salir del programa.
Llama a train_neuron, luego a visualize_training.
Imprime los resultados finales (épocas, pesos, bias).
Permite probar entradas personalizadas y ver la salida predicha.
