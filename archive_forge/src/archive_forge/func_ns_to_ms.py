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
def ns_to_ms(ns_time):
    return ns_time / NS_TO_MS_SCALE