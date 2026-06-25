import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import CartPoleEnv
from rnn import SimpleRNN
from esp import ESP


def experiment_population_size():
    """
    Эксперимент 1: Влияние размера субпопуляции на сходимость.
    """
    print("--- ГИПЕРПАРАМЕТР 1: Размер субпопуляции ---")
    N_HIDDEN = 5
    N_GENERATIONS = 30
    POP_SIZES = [10, 20, 50]
    
    env = CartPoleEnv(seed=SEED)
    rnn = SimpleRNN(env.n_inputs, N_HIDDEN, env.n_outputs)
    
    results = {}
    
    for pop_size in POP_SIZES:
        np.random.seed(SEED)
        esp = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=pop_size)
        history = []
        
        print(f"\nТест с pop_size={pop_size}:")
        for gen in range(N_GENERATIONS):
            esp.evaluate(env, rnn, episodes_per_ind=EPISODES_PER_IND)
            max_fit = np.max(esp.fitness)
            history.append(max_fit)
            esp.evolve()
            
            if gen % 5 == 0:
                print(f"  Gen {gen}: Max Fitness = {max_fit}")
        
        results[pop_size] = history
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    for idx, pop_size in enumerate(POP_SIZES):
        plt.plot(results[pop_size], label=f'POP_SIZE={pop_size}', 
                color=colors[idx], linewidth=2)
    
    plt.title('Hyperparameter Study: Population Size')
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparams_population_size.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env.close()


def experiment_mutation_std():
    """
    Эксперимент 2: Влияние силы мутации на сходимость.
    """
    print("\n--- ГИПЕРПАРАМЕТР 2: Сила мутации ---")
    N_HIDDEN = 5
    POP_SIZE = 20
    N_GENERATIONS = 30
    MUTATION_STDS = [0.1, 0.2, 0.5]
    
    env = CartPoleEnv(seed=SEED)
    rnn = SimpleRNN(env.n_inputs, N_HIDDEN, env.n_outputs)
    
    results = {}
    
    for mut_std in MUTATION_STDS:
        np.random.seed(SEED)
        esp = ESP(env.n_inputs, N_HIDDEN, env.n_outputs, pop_size=POP_SIZE)
        history = []
        
        print(f"\nТест с mutation_std={mut_std}:")
        for gen in range(N_GENERATIONS):
            esp.evaluate(env, rnn, episodes_per_ind=EPISODES_PER_IND)
            max_fit = np.max(esp.fitness)
            history.append(max_fit)
            esp.evolve(elite_size=ELITE_SIZE, mutation_std=mut_std)
            
            if gen % 5 == 0:
                print(f"  Gen {gen}: Max Fitness = {max_fit}")
        
        results[mut_std] = history
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    for idx, mut_std in enumerate(MUTATION_STDS):
        plt.plot(results[mut_std], label=f'MUTATION_STD={mut_std}', 
                color=colors[idx], linewidth=2)
    
    plt.title('Hyperparameter Study: Mutation Strength')
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparams_mutation_std.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env.close()


if __name__ == "__main__":
    experiment_population_size()
    experiment_mutation_std()