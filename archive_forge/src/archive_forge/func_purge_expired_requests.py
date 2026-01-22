import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def purge_expired_requests(self):
    self._purge()
    ray.autoscaler.sdk.request_resources(bundles=self._aggregate_requests())