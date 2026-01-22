import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def get_cpus(req):
    num_cpus = 0
    for r in req:
        if 'CPU' in r:
            num_cpus += r['CPU']
    return num_cpus