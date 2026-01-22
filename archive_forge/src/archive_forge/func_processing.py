import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
@ray.remote
def processing(delay):

    def sleep(x):
        print('Intermediate result', x)
        time.sleep(delay)
        return x
    ray.data.range(1000, parallelism=100).map(sleep, compute=ray.data.ActorPoolStrategy(size=1)).count()