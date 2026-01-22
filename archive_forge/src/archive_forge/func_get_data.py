import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
def get_data(self):
    if self.stopped:
        return {}
    with self.lock:
        ret_values = copy.deepcopy(self.values)
        for key, val in self.values.items():
            del val[:]
    return {'perf': {k: np.mean(v) for k, v in ret_values.items() if len(v) > 0}}