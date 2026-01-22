import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.examples.env.simple_corridor import SimpleCorridor
from ray.rllib.utils.test_utils import check_learning_achieved
def custom_eval_function(algorithm, eval_workers):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """
    eval_workers.foreach_worker(func=lambda w: w.foreach_env(lambda env: env.set_corridor_length(4 if w.worker_index == 1 else 7)))
    for i in range(5):
        print('Custom evaluation round', i)
        eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)
    episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    metrics = summarize_episodes(episodes)
    metrics['foo'] = 1
    return metrics