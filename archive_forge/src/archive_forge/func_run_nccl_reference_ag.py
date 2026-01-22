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
def run_nccl_reference_ag():
    torch.distributed.all_gather_into_tensor(gathered_input, scattered_input)
    for w, go in zip(weights, gathered_outputs_nccl_reference):
        torch.matmul(gathered_input, w, out=go)