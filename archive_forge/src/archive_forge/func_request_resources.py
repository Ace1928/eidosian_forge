import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def request_resources(self, req: List[Dict], execution_id: str):
    self._purge()
    self._resource_requests[execution_id] = (req, time.time() + self._timeout)
    ray.autoscaler.sdk.request_resources(bundles=self._aggregate_requests())