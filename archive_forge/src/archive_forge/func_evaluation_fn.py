import time
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1