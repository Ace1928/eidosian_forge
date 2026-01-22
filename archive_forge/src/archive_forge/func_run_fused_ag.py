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
def run_fused_ag():
    nonlocal gathered_outputs_fused
    from xformers.ops import fused_allgather_and_linear
    gathered_outputs_fused = fused_allgather_and_linear(scattered_input, [w.t() for w in weights], group=subgroup, num_stripes=2, timeout_s=10)