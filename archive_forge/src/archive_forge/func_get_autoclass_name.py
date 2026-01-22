import os
import subprocess
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Set
import numpy as np
import optuna
import torch
import transformers
from datasets import Dataset
from tqdm import trange
from . import version as optimum_version
from .utils.preprocessing import (
from .utils.runs import RunConfig, cpu_info_command
def get_autoclass_name(task):
    if task in ['text-classification', 'audio-classification']:
        autoclass_name = 'sequence-classification'
    else:
        autoclass_name = task
    return autoclass_name