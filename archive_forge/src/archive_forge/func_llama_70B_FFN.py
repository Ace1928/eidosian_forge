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
def llama_70B_FFN(world_size: int) -> Scenario:
    batch_size = world_size
    return Scenario(num_samples=batch_size * LLAMA_70B_SLEN, outer_dim=LLAMA_70B_D, inner_dim=round_up_to_nearest_multiple(2 * (4 * LLAMA_70B_D) // 3, 256) // world_size, num_ag_matrices=2)