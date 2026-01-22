import argparse
import os
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
def my_train_fn(config):
    iterations = config.pop('train-iterations', 10)
    config = PPOConfig().update_from_dict(config).environment('CartPole-v1')
    config.lr = 0.01
    agent1 = config.build()
    for _ in range(iterations):
        result = agent1.train()
        result['phase'] = 1
        train.report(result)
        phase1_time = result['timesteps_total']
    state = agent1.save()
    agent1.stop()
    config.lr = 0.0001
    agent2 = config.build()
    agent2.restore(state)
    for _ in range(iterations):
        result = agent2.train()
        result['phase'] = 2
        result['timesteps_total'] += phase1_time
        train.report(result)
    agent2.stop()