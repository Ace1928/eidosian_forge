import dataclasses
import os
from typing import Any, List
import torch
from .utils import print_once
def print_missing(stack):
    if any(('/torch/autograd/profiler.py' in x for x in stack)):
        return
    stack = [x for x in stack if '<built-in' not in x and 'site-packages/torch/' not in x]
    print_once('MISSING', ' >> '.join(stack[-3:]))