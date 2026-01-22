import time
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
def run_optuna_tune(smoke_test=False):
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    tuner = tune.Tuner(easy_objective, tune_config=tune.TuneConfig(metric='mean_loss', mode='min', search_alg=algo, scheduler=scheduler, num_samples=10 if smoke_test else 100), param_space={'steps': 100, 'width': tune.uniform(0, 20), 'height': tune.uniform(-100, 100), 'activation': tune.choice(['relu', 'tanh'])})
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)