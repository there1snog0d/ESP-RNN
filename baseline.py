import numpy as np
import pickle
import os

class StandardES:
    """
    Стандартный эволюционный алгоритм.
    
    В отличие от ESP, здесь каждая особь — это вся сеть целиком
    (один плоский вектор всех весов).
    """
    
    def __init__(self, n_inputs, n_hidden, n_outputs, pop_size=100):
        self.n_in = n_inputs
        self.n_hid = n_hidden
        self.n_out = n_outputs
        
        # Полный размер генома для ВСЕЙ сети
        # W_in (in*hid) + W_hid (hid*hid) + Bias (hid) + W_out (hid*out)
        self.genome_size = (n_inputs * n_hidden) + \
                           (n_hidden * n_hidden) + \
                           n_hidden + \
                           (n_hidden * n_outputs)
        
        self.pop_size = pop_size
        self.population = [np.random.uniform(-1, 1, self.genome_size) for _ in range(pop_size)]
        self.fitness = np.zeros(pop_size)
        
        self.best_genome = None
        self.best_score = -1

    def run_episode(self, env, genome, max_steps=500):
        """Прогон эпизода с использованием плоского (flat) генома."""
        obs = env.reset()
        h_prev = np.zeros(self.n_hid)
        total_reward = 0
        
        # Распаковываем плоский геном в матрицы весов
        idx = 0
        W_in = genome[idx : idx + self.n_in * self.n_hid].reshape(self.n_in, self.n_hid)
        idx += self.n_in * self.n_hid
        
        W_hid = genome[idx : idx + self.n_hid * self.n_hid].reshape(self.n_hid, self.n_hid)
        idx += self.n_hid * self.n_hid
        
        Bias = genome[idx : idx + self.n_hid]
        idx += self.n_hid
        
        W_out = genome[idx : idx + self.n_hid * self.n_out].reshape(self.n_hid, self.n_out)
        
        for _ in range(max_steps):
            # Векторизованный прямой проход
            h_new = np.tanh(np.dot(obs, W_in) + np.dot(h_prev, W_hid) + Bias)
            y = np.dot(h_new, W_out)
            action = np.argmax(y)
            
            obs, reward, done = env.step(action)
            total_reward += reward
            h_prev = h_new
            if done:
                break
        return total_reward

    def evaluate(self, env, episodes_per_ind=3, max_steps=500):
        """Оценивает каждую особь в нескольких эпизодах."""
        self.fitness = np.zeros(self.pop_size)
        self.best_score = -1
        
        for i in range(self.pop_size):
            total_reward = 0
            for _ in range(episodes_per_ind):
                reward = self.run_episode(env, self.population[i], max_steps)
                total_reward += reward
            
            avg_reward = total_reward / episodes_per_ind
            self.fitness[i] = avg_reward
            
            if avg_reward > self.best_score:
                self.best_score = avg_reward
                self.best_genome = self.population[i].copy()

    def evolve(self, elite_size=5, mutation_std=0.2):
        """Эволюция: селекция + мутация."""
        sorted_indices = np.argsort(self.fitness)[::-1]
        new_pop = []
        
        # Элитарность
        for i in range(elite_size):
            new_pop.append(self.population[sorted_indices[i]].copy())
            
        # Мутация
        for _ in range(elite_size, self.pop_size):
            parent_idx = sorted_indices[np.random.randint(0, elite_size)]
            child = self.population[parent_idx].copy()
            child += np.random.normal(0, mutation_std, self.genome_size)
            new_pop.append(child)
            
        self.population = new_pop

    def save(self, filename="baseline_es_model.pkl"):
        """Сохраняет лучшую сеть."""
        with open(filename, 'wb') as f:
            pickle.dump(self.best_genome, f)
        print(f"Baseline модель сохранена в {filename}")

    def load(self, filename="baseline_es_model.pkl"):
        """Загружает лучшую сеть из файла."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден!")
        with open(filename, 'rb') as f:
            self.best_genome = pickle.load(f)
        print(f"Baseline модель загружена из {filename}")