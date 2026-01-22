import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
Tracks information across the life of Tune loop and makes guesses
        about if Tune loop is stuck due to infeasible resources.
        If so, outputs certain warning messages.
        The logic should be conservative, non-intrusive and informative.
        For example, rate limiting is applied so that the message is not
        spammy.
        