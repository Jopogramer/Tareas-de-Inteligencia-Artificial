import numpy as np

#  Entorno personalizado
class CustomEnv:
    def __init__(self):
        self.state = 0
        self.goal = 10
        self.max_state = 20

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += action
        reward = 1 if self.state == self.goal else -0.1
        done = self.state >= self.goal
        return self.state, reward, done

#  Parámetros y configuración
actions = [1, 2]  # acciones posibles
num_states = 21
q_table = np.zeros((num_states, len(actions)))

alpha = 0.1       # tasa de aprendizaje
gamma = 0.9       # descuento futuro
epsilon = 1.0     # empieza completamente aleatorio
epsilon_min = 0.01
epsilon_decay = 0.99
episodes = 300

env = CustomEnv()

#  Bucle de entrenamiento (aprende desde política aleatoria)
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done and state < num_states:
        # Política epsilon-greedy (empieza aleatoria y mejora)
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.choice([0, 1])  # aleatorio
        else:
            action_idx = np.argmax(q_table[state])  # mejor acción conocida

        action = actions[action_idx]
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Actualización Q
        if next_state < num_states:
            old_value = q_table[state, action_idx]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action_idx] = new_value

        state = next_state

    # Disminuir epsilon (menos aleatorio con el tiempo)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Mostrar progreso cada 50 episodios
    if (ep + 1) % 50 == 0:
        print(f" Episodio {ep + 1} → Recompensa total: {total_reward:.2f}, ε = {epsilon:.3f}")

#  Resultados finales
print("\n Entrenamiento completado")
print(" Tabla Q final:")
for s in range(11):
    print(f"Estado {s:2d}: Acción 1 → {q_table[s][0]:.2f}, Acción 2 → {q_table[s][1]:.2f}")
