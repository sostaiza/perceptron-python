import numpy as np                # Librería para manejo de arreglos y operaciones matemáticas
import matplotlib.pyplot as plt   # Librería para graficar los resultados
from dataclasses import dataclass # Módulo para definir clases de datos inmutables

# Definición de la clase Perceptron utilizando dataclass
@dataclass(frozen=True)  # Esto hace que los objetos de la clase sean inmutables
class Perceptron:
    weights: np.ndarray  # Vector de pesos del perceptrón
    bias: float  # Sesgo (bias) del perceptrón

    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        """
        Función de activación Sigmoide: convierte cualquier valor en un número entre 0 y 1.
        Se usa para modelar la probabilidad de la salida.
        """
        return 1 / (1 + np.exp(-x))  # Fórmula de la función Sigmoide
    
    def predict(self, x: np.ndarray) -> int:
        """
        Predice la salida del perceptrón para una entrada dada.
        La salida es 1 si la activación es >= 0.5, de lo contrario es 0.
        """
        return int(self.activation(np.dot(x, self.weights) + self.bias) >= 0.5)

def train_neuron(gate_type: str, max_epochs=10_000, learning_rate=0.05, tolerance=1e-4, early_stop_threshold=200, delta_tolerance=1e-6, avg_window=50) -> tuple[Perceptron, list, int, np.ndarray, float]:
    """
    Entrena un perceptrón para aprender las compuertas lógicas AND y OR.
    Se usa la regla de actualización basada en el descenso del gradiente.

    Parámetros:
    - gate_type: Tipo de compuerta lógica a entrenar ("AND" o "OR").
    - max_epochs: Número máximo de épocas de entrenamiento.
    - learning_rate: Tasa de aprendizaje (qué tan grandes son los ajustes en cada iteración).
    - tolerance: Umbral de error mínimo para detener el entrenamiento.
    - early_stop_threshold: Cantidad de épocas sin mejora antes de detener el entrenamiento.
    - delta_tolerance: Cambios mínimos en la tasa de error promedio para detener el entrenamiento.
    - avg_window: Tamaño de la ventana para promediar el error en cada iteración.

    Retorna:
    - Un objeto Perceptron entrenado.
    - Lista con los errores en cada época.
    - Número de épocas necesarias para entrenar.
    - Pesos finales aprendidos.
    - Sesgo final aprendido.
    """

    # Definimos los datos de entrada y salida para las compuertas lógicas AND y OR
    logic_gates = {
        "AND": np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]),  # Tabla de verdad AND
        "OR": np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])   # Tabla de verdad OR
    }
    
    # Extraemos entradas (X) y salidas esperadas (y)
    X, y = logic_gates[gate_type][:, :2], logic_gates[gate_type][:, 2]

    # Inicializamos los pesos y el sesgo con valores aleatorios pequeños
    weights, bias = np.random.normal(0, 0.01, 2), np.random.normal(0, 0.01)

    # Variables para rastrear el error mínimo y mejores pesos/sesgo
    min_error, no_improvement_epochs, errors = np.inf, 0, []
    best_weights, best_bias, final_epoch = weights.copy(), bias, max_epochs
    
    # Bucle de entrenamiento
    for epoch in range(max_epochs):
        net_input = np.dot(X, weights) + bias  # Cálculo de la entrada neta (X * pesos + bias)
        output = Perceptron.activation(net_input)  # Aplicamos función de activación
        error = y - output  # Calculamos error
        total_error = np.sum(error ** 2)  # Error cuadrático total
        
        # Gradiente de la función de activación sigmoide
        gradient = error * output * (1 - output)

        # Actualización de pesos y bias usando regla de descenso del gradiente
        weights += learning_rate * np.dot(gradient, X)
        bias += learning_rate * np.sum(gradient)
        
        # Almacenamos el error
        errors.append(total_error)
        
        # Verificamos si mejoramos el error
        if total_error < min_error:
            min_error, best_weights, best_bias, no_improvement_epochs, final_epoch = total_error, weights.copy(), bias, 0, epoch + 1
        else:
            no_improvement_epochs += 1
        
        # Criterios de detención anticipada
        if no_improvement_epochs >= early_stop_threshold or total_error < tolerance or (len(errors) > avg_window and np.abs(errors[-1] - np.mean(errors[-avg_window:])) < delta_tolerance):
            print(f"Entrenamiento detenido en {final_epoch} épocas. Error final: {min_error:.6f}")
            break
    
    # Imprimimos los resultados finales
    print(f"Pesos finales: {best_weights}")
    print(f"Bias final: {best_bias}")
    print(f"Error final: {min_error:.6f}")
    
    # Devolvemos el perceptrón entrenado y los datos de entrenamiento
    return Perceptron(best_weights, best_bias), errors, final_epoch, best_weights, best_bias

def visualize_training(errors: list) -> None:
    """
    Genera un gráfico mostrando cómo evoluciona el error durante el entrenamiento.
    """
    plt.plot(errors)
    plt.xlabel('Épocas')
    plt.ylabel('Error')
    plt.title('Evolución del Error')
    plt.show()

def get_user_input() -> np.ndarray:
    """
    Solicita al usuario una entrada binaria (0 o 1) para probar el perceptrón.
    """
    while True:
        try:
            x = np.array(list(map(int, input("Ingrese entradas (0 o 1) separadas por espacio: ").split())))
            if np.all(np.isin(x, [0, 1])) and x.size == 2:
                return x
            print("Entradas inválidas. Deben ser 0 o 1.")
        except ValueError:
            print("Entrada no válida. Intente nuevamente.")

def main() -> None:
    """
    Función principal del programa.
    Permite al usuario seleccionar qué compuerta lógica entrenar y realizar predicciones.
    """
    try:
        options = {"1": "AND", "2": "OR"}
        
        # Menú interactivo para seleccionar la compuerta lógica
        while (option := input("\nSeleccione la compuerta lógica:\n1. AND\n2. OR\n3. Salir\nOpción: ")) != "3":
            if option not in options:
                print("Opción inválida. Intente de nuevo.")
                continue
            
            # Entrenamos el perceptrón
            perceptron, errors, epochs, weights, bias = train_neuron(options[option])

            # Mostramos la evolución del error en un gráfico
            visualize_training(errors)

            print(f"Entrenamiento completado en {epochs} épocas.")
            print(f"Pesos finales: {weights}")
            print(f"Bias final: {bias}")
            
            # Permitir al usuario probar el modelo con entradas personalizadas
            while True:
                x = get_user_input()
                print(f"Salida predicha: {perceptron.predict(x)}")
                if input("¿Probar otra entrada? (s/n): ").lower() != "s":
                    break
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario. Saliendo...")

# Ejecutar el programa solo si este archivo es el principal
if __name__ == "__main__":
    main()
