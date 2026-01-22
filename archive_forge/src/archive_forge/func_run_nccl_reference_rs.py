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
def run_nccl_reference_rs():
    for w, go, so in zip(weights, gathered_outputs, scattered_outputs_nccl_reference):
        torch.matmul(gathered_input, w, out=go)
        torch.distributed.reduce_scatter_tensor(so, go)