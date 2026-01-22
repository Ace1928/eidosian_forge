import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
def default_hp_space_wandb(trial) -> Dict[str, float]:
    from .integrations import is_wandb_available
    if not is_wandb_available():
        raise ImportError('This function needs wandb installed: `pip install wandb`')
    return {'method': 'random', 'metric': {'name': 'objective', 'goal': 'minimize'}, 'parameters': {'learning_rate': {'distribution': 'uniform', 'min': 1e-06, 'max': 0.0001}, 'num_train_epochs': {'distribution': 'int_uniform', 'min': 1, 'max': 6}, 'seed': {'distribution': 'int_uniform', 'min': 1, 'max': 40}, 'per_device_train_batch_size': {'values': [4, 8, 16, 32, 64]}}}