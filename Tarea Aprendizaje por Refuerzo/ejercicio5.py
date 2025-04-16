import numpy as np

class RestrictedEnv:
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Acción restringida (no permitida)
        if action == -1:
            print("⚠️ Acción prohibida detectada.")
            return self.state, -10, True  # castigo y finaliza episodio
        # Acción permitida
        self.state += action
        reward = 1 if self.state == 5 else -0.1
        done = self.state >= 5
        return self.state, reward, done

# Crear entorno
env = RestrictedEnv()

# Política simple: nunca elige acción prohibida
acciones_posibles = [1, 2]  # solo permitidas

# Ejecutar varios episodios
num_episodios = 5

for episodio in range(num_episodios):
    state = env.reset()
    done = False
    total_reward = 0
    print(f"\n Episodio {episodio + 1} iniciado")
    
    while not done:
        # Acción aleatoria pero dentro de las permitidas
        action = np.random.choice(acciones_posibles)
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f" → Estado: {next_state}, Acción: {action}, Recompensa: {reward}")
    
    print(f" Episodio {episodio + 1} terminado con recompensa total: {total_reward:.2f}")
