import numpy as np
from collections import defaultdict

# Política epsilon-greedy: exploración/explotación
def elegir_accion(q, estado, acciones, epsilon=0.1):
    """Política epsilon-greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(acciones)  # Exploración: tomar una acción aleatoria
    else:
        valores = [q[(estado, a)] for a in acciones]  # Explotación: tomar la mejor acción
        return acciones[np.argmax(valores)]

def monte_carlo_control(episodios=1000, epsilon=0.1, acciones=None):
    """
    Algoritmo Monte Carlo Control para resolver un problema de Aprendizaje por Refuerzo Continuo
    como el control de un brazo robótico que debe llegar a una posición objetivo (0).
    """
    if acciones is None:
        acciones = [-0.1, 0, 0.1]  # Movimientos discretos que puede hacer el brazo
    
    q = defaultdict(float)  # Q-Tabla para almacenar las estimaciones de valor
    retornos = defaultdict(list)  # Para almacenar las recompensas obtenidas en episodios

    # Simulación de episodios de entrenamiento
    for episodio in range(episodios):
        # Inicialización del episodio: estado es una posición aleatoria en el rango [-1, 1]
        estado = round(np.random.uniform(-1, 1), 1)
        episodio_pasos = []

        # Simulación de la trayectoria del episodio
        while abs(estado) > 0.1:  # El brazo sigue moviéndose hasta estar cerca de la posición objetivo
            accion = elegir_accion(q, estado, acciones, epsilon)  # Elegir una acción
            siguiente_estado = round(estado + accion, 1)  # Calcular el siguiente estado
            recompensa = -abs(siguiente_estado)  # Recompensa: penalización por estar lejos del objetivo

            episodio_pasos.append((estado, accion, recompensa))  # Guardar el paso del episodio
            estado = siguiente_estado  # Actualizar el estado actual

        # Actualización de Q para cada (estado, acción) en el episodio
        G = 0  # Variable para la suma de recompensas futuras (retorno)
        for estado, accion, recompensa in reversed(episodio_pasos):
            G = recompensa + G  # Sumar las recompensas del futuro
            clave = (estado, accion)
            retornos[clave].append(G)  # Guardar los retornos
            q[clave] = np.mean(retornos[clave])  # Actualizar Q estimado
        
        # Mostrar progreso cada 100 episodios
        if episodio % 100 == 0:
            print(f"Progreso: Episodio {episodio}/{episodios}")

    return q

# Ejecutar el aprendizaje
q_estimado = monte_carlo_control(episodios=500)  # Reducir el número de episodios para pruebas rápidas

# Mostrar la Q estimada después del entrenamiento
print("\n Aprendizaje completado. Q estimado para el brazo robótico:")
for clave, valor in sorted(q_estimado.items()):
    print(f"Estado {clave[0]:>5}, Acción {clave[1]:>4} → Q ≈ {valor:.3f}")
