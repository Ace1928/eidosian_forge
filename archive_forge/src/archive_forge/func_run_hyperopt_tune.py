import time
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
def run_hyperopt_tune(config_dict=config_space, smoke_test=False):
    algo = HyperOptSearch(space=config_dict, metric='mean_loss', mode='min')
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    tuner = tune.Tuner(easy_objective, tune_config=tune.TuneConfig(metric='mean_loss', mode='min', search_alg=algo, scheduler=scheduler, num_samples=10 if smoke_test else 100))
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)