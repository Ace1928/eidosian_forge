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
def run_compute_lower_bound_rs():
    for w, go, so in zip(weights, gathered_outputs, scattered_outputs):
        torch.matmul(gathered_input, w, out=go)
        torch.sum(go.view((world_size, M // world_size, N)), dim=0, out=so)