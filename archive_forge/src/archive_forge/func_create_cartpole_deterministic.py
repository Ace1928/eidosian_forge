import gymnasium as gym
def create_cartpole_deterministic(config):
    env = gym.make('CartPole-v1')
    env.reset(seed=config.get('seed', 0))
    return env