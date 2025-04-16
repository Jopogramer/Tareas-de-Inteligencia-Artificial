import numpy as np

# Definir los agentes con puntaje inicial en 0
agents = {"A": 0, "B": 0}

# Simulación de 10 turnos
for turno in range(10):
    print(f"Turno {turno + 1}:")
    for name in agents:
        action = np.random.choice([1, 2])  # Acción aleatoria: 1 o 2 puntos
        agents[name] += action
        print(f"  Agente {name} realiza acción {action} → Total: {agents[name]}")
    print()

# Mostrar puntajes finales
print("Puntajes finales: ")
for name, score in agents.items():
    print(f"Agente {name}: {score} puntos")
