import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import CartPoleEnv, NoisyCartPoleEnv
from rnn import SimpleRNN
from esp import ESP


def ablation_study():
    """
    Механистическая абляция: влияние количества эпизодов оценки на сходимость ESP.
    """
    print("--- ЭКСПЕРИМЕНТ 1: АБЛЯЦИЯ (Влияние episodes_per_ind) ---")
    N_HIDDEN = 5
    POP_SIZE = 20
    N_GENERATIONS = 30
    
    env = CartPoleEnv(seed=SEED)
    rnn = SimpleRNN(env.n_inputs, N_HIDDEN, env.n_outputs)
    
    results = {}
    
    for episodes in ABLATION_EPISODES:
        np.random.seed(SEED)
        esp = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=POP_SIZE)
        history = []
        
        print(f"\nТест с episodes_per_ind={episodes}:")
        for gen in range(N_GENERATIONS):
            esp.evaluate(env, rnn, episodes_per_ind=episodes)
            max_fit = np.max(esp.fitness)
            history.append(max_fit)
            esp.evolve()
            
            if gen % 5 == 0:
                print(f"  Gen {gen}: Max Fitness = {max_fit}")
        
        results[episodes] = history
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green']
    labels = ['ESP (1 episode per ind) - Broken', 'ESP (3 episodes per ind) - Normal']
    
    for idx, episodes in enumerate(ABLATION_EPISODES):
        plt.plot(results[episodes], label=labels[idx], color=colors[idx], linewidth=2, 
                linestyle='--' if episodes == 1 else '-')
    
    plt.title('Ablation Study: Impact of Evaluation Noise')
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env.close()


def robustness_test():
    """
    Тест устойчивости RNN к шуму в наблюдениях.
    """
    print("\n--- ЭКСПЕРИМЕНТ 2: УСТОЙЧИВОСТЬ (RNN против Шума) ---")
    N_HIDDEN = 5
    POP_SIZE = 20
    N_GENERATIONS = 50
    
    # 1. Обучаем сеть в чистой среде
    env_clean = CartPoleEnv(seed=SEED)
    rnn = SimpleRNN(env_clean.n_inputs, N_HIDDEN, env_clean.n_outputs)
    esp = ESP(env_clean.n_inputs, N_HIDDEN, env_clean.n_outputs, pop_size=POP_SIZE)
    
    print("Обучение в чистой среде...")
    for gen in range(N_GENERATIONS):
        esp.evaluate(env_clean, rnn, episodes_per_ind=EPISODES_PER_IND)
        if esp.best_network_score >= SUCCESS_THRESHOLD:
            print(f"Решение найдено на поколении {gen}!")
            break
        esp.evolve()
        
    best_net = esp.best_network
    print(f"Обучение завершено. Лучший фитнес: {esp.best_network_score}")
    
    # 2. Тестируем на разных уровнях шума
    scores = []
    for noise in NOISE_LEVELS:
        env_noisy = NoisyCartPoleEnv(seed=SEED, noise_std=noise)
        total_reward = 0
        for _ in range(5):
            total_reward += rnn.run_episode(env_noisy, best_net, max_steps=MAX_STEPS)
        avg_score = total_reward / 5
        scores.append(avg_score)
        print(f"Шум {noise:.2f} -> Средняя награда: {avg_score:.1f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(NOISE_LEVELS, scores, marker='o', color='blue', linewidth=2, markersize=8)
    plt.title('Robustness Test: RNN Performance under Observation Noise')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Average Steps Survived')
    plt.grid(True)
    plt.savefig('robustness_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env_clean.close()


def robustness_with_augmentation():
    """
    Обучение с шумом (Data Augmentation) для повышения устойчивости.
    """
    print("\n--- ЭКСПЕРИМЕНТ 3: ОБУЧЕНИЕ С ШУМОМ (Data Augmentation) ---")
    N_HIDDEN = 5
    POP_SIZE = 20
    N_GENERATIONS = 50
    TRAIN_NOISE = 0.1
    
    # 1. Обучаем сеть СРАЗУ в зашумленной среде
    env_noisy_train = NoisyCartPoleEnv(seed=SEED, noise_std=TRAIN_NOISE)
    rnn = SimpleRNN(env_noisy_train.n_inputs, N_HIDDEN, env_noisy_train.n_outputs)
    esp = ESP(env_noisy_train.n_inputs, N_HIDDEN, env_noisy_train.n_outputs, pop_size=POP_SIZE)
    
    print(f"Обучение в ЗАШУМЛЕННОЙ среде (noise={TRAIN_NOISE})...")
    for gen in range(N_GENERATIONS):
        esp.evaluate(env_noisy_train, rnn, episodes_per_ind=EPISODES_PER_IND)
        max_fit = np.max(esp.fitness)
        if gen % 10 == 0:
            print(f"  Gen {gen}: Max Fitness = {max_fit}")
        if max_fit >= SUCCESS_THRESHOLD:
            print(f"Решение найдено на поколении {gen}!")
            break
        esp.evolve()
        
    best_net_aug = esp.best_network
    print(f"Обучение завершено. Лучший фитнес: {esp.best_network_score}")
    
    # 2. Тестируем на разных уровнях шума
    scores_aug = []
    for noise in NOISE_LEVELS:
        env_test = NoisyCartPoleEnv(seed=SEED, noise_std=noise)
        total_reward = 0
        for _ in range(5):
            total_reward += rnn.run_episode(env_test, best_net_aug, max_steps=MAX_STEPS)
        avg_score = total_reward / 5
        scores_aug.append(avg_score)
        print(f"Тест Шум {noise:.2f} -> Средняя награда: {avg_score:.1f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(NOISE_LEVELS, scores_aug, marker='s', color='red', linewidth=2, 
             markersize=8, label='RNN Trained WITH Noise')
    plt.title('Robustness: Training with Noise (Data Augmentation)')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Average Steps Survived')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_with_augmentation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env_noisy_train.close()


if __name__ == "__main__":
    ablation_study()
    robustness_test()
