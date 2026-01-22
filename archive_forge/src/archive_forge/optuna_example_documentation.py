import time
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
This example demonstrates the usage of Optuna with Ray Tune.

It also checks that it is usable with a separate scheduler.

Requires the Optuna library to be installed (`pip install optuna`).

For an example of using an Optuna define-by-run function, see
:doc:`/tune/examples/optuna_define_by_run_example`.
