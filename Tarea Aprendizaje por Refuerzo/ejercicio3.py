import random

# Q-learning para el problema de la mochila
def q_learning(capacidad, pesos, valores, num_episodios=1000, tasa_aprendizaje=0.1, descuento=0.9, epsilon=0.1):
    n = len(pesos)
    
    # Inicialización de la Q-table (tamaño: (n+1) x (capacidad+1)), con valores en 0
    Q = [[0 for _ in range(capacidad + 1)] for _ in range(n + 1)]
    
    # Función para seleccionar una acción (con exploración y explotación)
    def seleccionar_accion(s):
        if random.uniform(0, 1) < epsilon:
            return random.choice([0, 1])  # Acción aleatoria (exploración)
        else:
            return max(range(2), key=lambda a: Q[s][a])  # Mejor acción (explotación)

    # Entrenamiento
    for episodio in range(num_episodios):
        estado = (0, capacidad)  # Empezamos sin elementos seleccionados y capacidad completa
        
        while estado[0] < n:
            # Seleccionamos una acción: 0 = no tomar el objeto, 1 = tomar el objeto
            accion = seleccionar_accion(estado[0])
            
            if accion == 1 and pesos[estado[0]] <= estado[1]:  # Si el objeto cabe en la mochila
                nueva_recompensa = valores[estado[0]]
                nueva_estado = (estado[0] + 1, estado[1] - pesos[estado[0]])
            else:
                nueva_recompensa = 0
                nueva_estado = (estado[0] + 1, estado[1])  # No tomamos el objeto, pero avanzamos al siguiente

            # Actualización de la Q-table con la fórmula de Q-learning
            Q[estado[0]][accion] += tasa_aprendizaje * (
                nueva_recompensa + descuento * max(Q[nueva_estado[0]]) - Q[estado[0]][accion]
            )
            
            estado = nueva_estado

    # Después del entrenamiento, devolvemos la Q-table
    return Q

# Ejemplo de uso
valores = [60, 100, 120]
pesos = [10, 20, 30]
capacidad = 50

# Entrenamos el modelo de Q-learning
Q = q_learning(capacidad, pesos, valores)

# Ver la Q-table resultante
for i in range(len(pesos)):
    print(f"Q({i}, capacidad) = {Q[i]}")
