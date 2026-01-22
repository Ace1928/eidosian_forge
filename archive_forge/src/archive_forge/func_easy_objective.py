import time
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
def easy_objective(config):
    width, height = (config['width'], config['height'])
    for step in range(config['steps']):
        intermediate_score = evaluation_fn(step, width, height)
        train.report({'iterations': step, 'mean_loss': intermediate_score})
        time.sleep(0.1)