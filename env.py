import gymnasium as gym
import numpy as np

class CartPoleEnv:
    """Базовая среда CartPole-v1."""
    
    def __init__(self, seed=42):
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        
        self.n_inputs = self.env.observation_space.shape[0]  # 4
        self.n_outputs = self.env.action_space.n             # 2

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def close(self):
        self.env.close()

class NoisyCartPoleEnv:
    """Среда CartPole-v1 с гауссовым шумом в наблюдениях."""
    
    def __init__(self, seed=42, noise_std=0.1):
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = self.env.action_space.n
        self.noise_std = noise_std

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return obs + np.random.normal(0, self.noise_std, size=obs.shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        noisy_obs = obs + np.random.normal(0, self.noise_std, size=obs.shape)
        return noisy_obs, reward, done

    def close(self):
        self.env.close()