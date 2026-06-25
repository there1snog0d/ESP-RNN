import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import CartPoleEnv
from baseline import StandardES


def train_baseline():
    """Обучение Standard ES на среде CartPole-v1."""
    np.random.seed(SEED)
    env = CartPoleEnv(seed=SEED)
    es = StandardES(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=BASELINE_POP_SIZE)
    
    history_best = []
    history_avg = []
    
    print(f"Запуск Baseline (Standard ES) на {N_GENERATIONS} поколений...")
    print(f"Параметры: POP_SIZE={BASELINE_POP_SIZE}, EPISODES_PER_IND={EPISODES_PER_IND}")
    print("Поколение | Лучший фитнес | Средний фитнес")
    print("-" * 50)
    
    for gen in range(N_GENERATIONS):
        es.evaluate(env, episodes_per_ind=EPISODES_PER_IND)
        
        max_fitness = np.max(es.fitness)
        avg_fitness = np.mean(es.fitness)
        history_best.append(max_fitness)
        history_avg.append(avg_fitness)
        
        if gen % 5 == 0 or gen == N_GENERATIONS - 1:
            print(f"Gen {gen:3d}   | {max_fitness:10.2f} | {avg_fitness:10.2f}")
            
        if es.best_score >= SUCCESS_THRESHOLD:
            print(f"\nBASELINE РЕШЕНА на поколении {gen}! Лучший фитнес: {es.best_score}")
            break
            
        es.evolve(elite_size=BASELINE_ELITE_SIZE, mutation_std=BASELINE_MUTATION_STD)

    es.save("baseline_es_model.pkl")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(history_best, label='Best Fitness', color='red', linewidth=2)
    plt.plot(history_avg, label='Average Fitness', color='orange', linewidth=2)
    plt.title('Baseline (Standard ES) Training Progress (CartPole-v1)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Steps survived)')
    plt.legend()
    plt.grid(True)
    plt.savefig('baseline_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Тестирование загруженной модели
    print("\n--- Тестирование загруженной модели ---")
    es_test = StandardES(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=BASELINE_POP_SIZE)
    es_test.load("baseline_es_model.pkl")
    
    # Тестируем на 5 эпизодах
    total_reward = 0
    for i in range(5):
        reward = es_test.run_episode(env, es_test.best_genome, max_steps=MAX_STEPS)
        total_reward += reward
        print(f"Эпизод {i+1}: Награда = {reward}")
        
    print(f"\nСредняя награда за 5 эпизодов: {total_reward/5:.1f}")
    
    env.close()
    return history_best, history_avg


if __name__ == "__main__":
    train_baseline()