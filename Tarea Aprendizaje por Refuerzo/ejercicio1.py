import gym
import numpy as np

# Crear entorno Taxi-v3
env = gym.make("Taxi-v3")

# Inicializar la tabla Q
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hiperparámetros
alpha = 0.1    
gamma = 0.6     
epsilon = 0.1  
episodes = 1000

# Para seguimiento de recompensas
rewards_per_episode = []

print(" Iniciando entrenamiento Q-Learning en Taxi-v3...")

for i in range(1, episodes + 1):
    state = env.reset()[0]
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Selección de acción (ε-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Ejecutar acción
        next_state, reward, done, _, _ = env.step(action)

        # Actualizar tabla Q
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward
        steps += 1

    rewards_per_episode.append(total_reward)

    # Mostrar información cada 100 episodios
    if i % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"🧾 Episodio {i} - Recompensa media últimos 100 episodios: {avg_reward:.2f}")

print("\n Entrenamiento completado")
print(f" Recompensa media final (últimos 100): {np.mean(rewards_per_episode[-100:]):.2f}")
print(f" Mejor recompensa alcanzada en un episodio: {np.max(rewards_per_episode)}")
print(f" Peor recompensa: {np.min(rewards_per_episode)}")

# Cerrar el entorno
env.close()
