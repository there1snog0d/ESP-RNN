import numpy as np
import pickle
import os

class ESP:
    """
    Алгоритм ESP для обучения рекуррентной нейронной сети.
    
    Ключевая идея: веса сети разбиваются на группы (субпопуляции).
    Каждая субпопуляция отвечает за входящие веса одного конкретного скрытого нейрона.
    """
    
    def __init__(self, n_inputs, n_hidden, n_outputs, pop_size=20):
        self.n_hid = n_hidden
        self.pop_size = pop_size
        self.genome_size = n_inputs + n_hidden + 1 + n_outputs
        
        # Инициализация субпопуляций случайными весами
        self.subpopulations = [
            [np.random.uniform(-1, 1, self.genome_size) for _ in range(pop_size)]
            for _ in range(n_hidden)
        ]
        
        # Матрица фитнеса: [субпопуляция, особь]
        self.fitness = np.zeros((n_hidden, pop_size))
        
        # Отслеживание лучшей полной сети
        self.best_network = None
        self.best_network_score = -1

    def evaluate(self, env, rnn, episodes_per_ind=3, max_steps=500):
        """
        Оценивает каждую особь, собирая её со случайными напарниками.
        
        Кумулятивная оценка: каждая особь оценивается в нескольких эпизодах
        с разными случайными напарниками из других субпопуляций.
        """
        self.fitness = np.zeros((self.n_hid, self.pop_size))
        self.best_network_score = -1
        
        for i in range(self.n_hid):
            for j in range(self.pop_size):
                for _ in range(episodes_per_ind):
                    genome_list = []
                    
                    for k in range(self.n_hid):
                        if k == i:
                            # Берём текущую оцениваемую особь
                            genome_list.append(self.subpopulations[k][j])
                        else:
                            # Берём случайную особь из другой субпопуляции
                            rand_idx = np.random.randint(0, self.pop_size)
                            genome_list.append(self.subpopulations[k][rand_idx])
                    
                    # Прогоняем сеть и накапливаем фитнес
                    reward = rnn.run_episode(env, genome_list, max_steps)
                    self.fitness[i, j] += reward
                    
                    # Запоминаем лучшую собранную сеть
                    if reward > self.best_network_score:
                        self.best_network_score = reward
                        self.best_network = [g.copy() for g in genome_list]
                        
        # Усредняем фитнес
        self.fitness /= episodes_per_ind

    def evolve(self, elite_size=3, mutation_std=0.2, crossover_prob=0.7):
        for i in range(self.n_hid):
            # Сортируем особей по фитнесу (от лучшего к худшему)
            sorted_indices = np.argsort(self.fitness[i])[::-1]
            
            new_subpop = []
            
            # Элитарность: лучшие переходят в новое поколение как есть
            for j in range(elite_size):
                best_idx = sorted_indices[j]
                new_subpop.append(self.subpopulations[i][best_idx].copy())
                
            #  Скрещивание и мутация
            j = elite_size
            while j < self.pop_size:
                # Выбираем двух родителей из элиты (турнирный отбор из топ-50%)
                parent1_idx = sorted_indices[np.random.randint(0, max(elite_size, self.pop_size // 2))]
                parent2_idx = sorted_indices[np.random.randint(0, max(elite_size, self.pop_size // 2))]
                
                parent1 = self.subpopulations[i][parent1_idx].copy()
                parent2 = self.subpopulations[i][parent2_idx].copy()
                
                # Применяем скрещивание с вероятностью crossover_prob
                if np.random.random() < crossover_prob:
                    # Одноточечное скрещивание
                    crossover_point = np.random.randint(1, self.genome_size)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                else:
                    # Без скрещивания - просто копируем родителей
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Применяем мутацию к обоим потомкам
                mutation1 = np.random.normal(0, mutation_std, size=self.genome_size)
                mutation2 = np.random.normal(0, mutation_std, size=self.genome_size)
                child1 += mutation1
                child2 += mutation2
                
                new_subpop.append(child1)
                if j + 1 < self.pop_size:
                    new_subpop.append(child2)
                
                j += 2
                
            # Обрезаем до нужного размера (если добавили лишнего)
            new_subpop = new_subpop[:self.pop_size]
            
            # Заменяем старую субпопуляцию новой
            self.subpopulations[i] = new_subpop

    def save(self, filename="esp_model.pkl"):
        """Сохраняет лучшую сеть в файл."""
        if self.best_network is None:
            raise ValueError("Сначала нужно провести оценку (evaluate), чтобы сохранить лучшую сеть!")
        with open(filename, 'wb') as f:
            pickle.dump(self.best_network, f)
        print(f"Сохранена лучшая сеть с рекордом {self.best_network_score} шагов в {filename}")

    def load(self, filename="esp_model.pkl"):
        """Загружает сеть из файла."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден!")
        with open(filename, 'rb') as f:
            self.best_network = pickle.load(f)
        print(f"Модель загружена из {filename}")

    def get_best_genome_list(self):
        """Возвращает лучшую найденную сеть."""
        return self.best_network