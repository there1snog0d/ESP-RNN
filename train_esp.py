import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import CartPoleEnv
from rnn import SimpleRNN
from esp import ESP


def train_esp():
    """Обучение ESP на среде CartPole-v1."""
    np.random.seed(SEED)
    env = CartPoleEnv(seed=SEED)
    rnn = SimpleRNN(env.n_inputs, N_HIDDEN, env.n_outputs)
    esp = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=POP_SIZE)
    
    history_best = []
    history_avg = []
    
    print(f"Запуск эволюции ESP на {N_GENERATIONS} поколений...")
    print(f"Параметры: N_HIDDEN={N_HIDDEN}, POP_SIZE={POP_SIZE}, EPISODES_PER_IND={EPISODES_PER_IND}")
    print("Поколение | Лучший фитнес | Средний фитнес")
    print("-" * 50)
    
    for gen in range(N_GENERATIONS):
        # Оценка
        esp.evaluate(env, rnn, episodes_per_ind=EPISODES_PER_IND)
        
        # Статистика
        max_fitness = np.max(esp.fitness)
        avg_fitness = np.mean(esp.fitness)
        history_best.append(max_fitness)
        history_avg.append(avg_fitness)
        
        if gen % 5 == 0 or gen == N_GENERATIONS - 1:
            print(f"Gen {gen:3d}   | {max_fitness:10.2f} | {avg_fitness:10.2f}")
            
        # Проверка на решение
        if max_fitness >= SUCCESS_THRESHOLD:
            print(f"\nЗАДАЧА РЕШЕНА на поколении {gen}! Максимальный фитнес: {max_fitness}")
            break
            
        # Эволюция
        esp.evolve(elite_size=ELITE_SIZE, mutation_std=MUTATION_STD)

    # Сохранение модели
    esp.save("esp_model.pkl")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(history_best, label='Best Fitness', color='green', linewidth=2)
    plt.plot(history_avg, label='Average Fitness', color='blue', linewidth=2)
    plt.title('ESP Training Progress (CartPole-v1)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Steps survived)')
    plt.legend()
    plt.grid(True)
    plt.savefig('esp_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Тестирование загруженной модели
    print("\n--- Тестирование загруженной модели ---")
    esp_test = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=POP_SIZE)
    esp_test.load("esp_model.pkl")
    
    best_genome_list = esp_test.get_best_genome_list()
    
    total_reward = 0
    for i in range(5):
        reward = rnn.run_episode(env, best_genome_list, max_steps=MAX_STEPS)
        total_reward += reward
        print(f"Эпизод {i+1}: Награда = {reward}")
        
    print(f"\nСредняя награда за 5 эпизодов: {total_reward/5:.1f}")
      # ===== ВИЗУАЛИЗАЦИЯ =====
    print("\n" + "="*60)
    print("Генерация визуализаций...")
    print("="*60)
    
    from visualize import (create_network_visualization, 
                          create_genome_statistics,)
    
    # Загружаем лучшую модель для визуализации
    esp_test = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=POP_SIZE)
    esp_test.load("esp_model.pkl")
    best_genome_list = esp_test.get_best_genome_list()
    
    # 1. Визуализация структуры сети
    create_network_visualization(
        best_genome_list, 
        env.n_inputs, 
        N_HIDDEN, 
        env.n_outputs,
        filename='network_structure.png',
        title=f'RNN Structure (Best Solution)\nInputs: {env.n_inputs}, Hidden: {N_HIDDEN}, Outputs: {env.n_outputs}'
    )
    
    # 2. Статистика геномов
    create_genome_statistics(
        best_genome_list,
        env.n_inputs,
        N_HIDDEN,
        env.n_outputs,
        filename='genome_statistics.png'
    )
    
    env.close()
    return history_best, history_avg


if __name__ == "__main__":
    history_best, history_avg = train_esp()