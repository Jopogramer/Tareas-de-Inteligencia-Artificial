import numpy as np

# Definir los agentes con tablas Q separadas
agents = {"A": np.zeros((6, 2)), "B": np.zeros((6, 2))}
actions = [1, 2]
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
episodes = 300

# Simulación de aprendizaje multiagente con competencia
for ep in range(episodes):
    states = {"A": 0, "B": 0}
    done = {"A": False, "B": False}
    total_rewards = {"A": 0, "B": 0}

    while not all(done.values()):
        for agent in agents:
            if done[agent]:
                continue

            # Selección de acción (ε-greedy)
            q_table = agents[agent]
            state = states[agent]
            if np.random.uniform(0, 1) < epsilon:
                action_idx = np.random.choice([0, 1])
            else:
                action_idx = np.argmax(q_table[state])

            action = actions[action_idx]
            next_state = state + action

            # Competencia: si ambos agentes llegan al mismo estado, ninguno recibe recompensa
            if next_state == states["A"] and next_state == states["B"]:
                reward = 0  # Ningún agente recibe recompensa
            else:
                reward = 1 if next_state == 5 else -0.1

            done[agent] = next_state >= 5
            total_rewards[agent] += reward

            # Actualización Q
            if next_state < 6:
                old_value = q_table[state, action_idx]
                next_max = np.max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action_idx] = new_value

            states[agent] = next_state

    # Disminuir epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Mostrar progreso
    if (ep + 1) % 50 == 0:
        print(f"Episodio {ep + 1} → Recompensas: {total_rewards}, ε = {epsilon:.3f}")

print("\nEntrenamiento multiagente completado")