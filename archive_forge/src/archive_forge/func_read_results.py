import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap
import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List
def read_results(result_file: str):
    output = []
    with open(result_file, 'rb') as f:
        while True:
            try:
                output.append(pickle.load(f))
            except EOFError:
                break
    return output