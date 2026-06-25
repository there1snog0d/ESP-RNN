import numpy as np

class SimpleRNN:
    """
    RNN с 1 скрытым слоем.
    
    Архитектура:
    - Входной слой: n_inputs нейронов
    - Скрытый слой: n_hidden нейронов (полносвязный, рекуррентный)
    - Выходной слой: n_outputs нейронов
    
    Функция активации скрытого слоя: tanh
    """
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_in = n_inputs
        self.n_hid = n_hidden
        self.n_out = n_outputs
        
        # Размер генома одной особи (одного скрытого нейрона) в ESP:
        # Веса от входов (n_in) + Веса от всех скрытых (n_hid) + Bias (1) + Веса к выходам (n_out)
        self.genome_size = n_inputs + n_hidden + 1 + n_outputs

    def forward(self, x, h_prev, genome_list):
        """
        Прямой проход RNN.
        
        Args:
            x: текущий вход из среды (вектор размерности n_in)
            h_prev: предыдущее состояние скрытого слоя (вектор размерности n_hid)
            genome_list: список из n_hid массивов (весов). Каждый массив - геном одного нейрона.
            
        Returns:
            action: выбранное действие (int)
            h_new: новое состояние скрытого слоя (вектор размерности n_hid)
        """
        h_new = np.zeros(self.n_hid)
        y = np.zeros(self.n_out)
        
        for i in range(self.n_hid):
            g = genome_list[i]
            idx = 0
            
            # Распаковываем геном на составляющие
            w_in = g[idx : idx + self.n_in]; idx += self.n_in
            w_hid = g[idx : idx + self.n_hid]; idx += self.n_hid
            bias = g[idx]; idx += 1
            w_out = g[idx : idx + self.n_out]
            
            # Суммируем сигналы для i-го нейрона
            val = np.dot(w_in, x) + np.dot(w_hid, h_prev) + bias
            
            # Функция активации (tanh лучше для рекуррентных сетей)
            h_new[i] = np.tanh(val)
            
            # Накопливаем вклад этого нейрона в выходы
            y += h_new[i] * w_out

        # Выбираем действие (argmax по выходам)
        action = np.argmax(y)
        
        return action, h_new

    def run_episode(self, env, genome_list, max_steps=500):
        """
        Прогоняет один эпизод в среде и возвращает сумму наград.
        
        Args:
            env: среда Gymnasium
            genome_list: список геномов (по одному на каждый скрытый нейрон)
            max_steps: максимальное количество шагов
            
        Returns:
            total_reward: сумма наград за эпизод
        """
        obs = env.reset()
        h_prev = np.zeros(self.n_hid)
        total_reward = 0
        
        for _ in range(max_steps):
            action, h_prev = self.forward(obs, h_prev, genome_list)
            obs, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
                
        return total_reward