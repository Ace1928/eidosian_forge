import argparse
import contextlib
import dataclasses
import enum
import multiprocessing
import os
import random
from collections import deque
from statistics import mean, stdev
from typing import Callable
import torch
def run_fused_nowait_rs():
    nonlocal scattered_outputs_fused
    from xformers.ops import fused_linear_and_reducescatter
    scattered_outputs_fused = fused_linear_and_reducescatter(gathered_input, [w.t() for w in weights], group=subgroup_nowait, num_stripes=2, _wait=False, timeout_s=10)