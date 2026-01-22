import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_or_create_autoscaling_requester_actor():
    ctx = DataContext.get_current()
    scheduling_strategy = ctx.scheduling_strategy
    scheduling_strategy = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=True, _spill_on_unavailable=True)
    with _autoscaling_requester_lock:
        return AutoscalingRequester.options(name='AutoscalingRequester', namespace='AutoscalingRequester', get_if_exists=True, lifetime='detached', scheduling_strategy=scheduling_strategy).remote()