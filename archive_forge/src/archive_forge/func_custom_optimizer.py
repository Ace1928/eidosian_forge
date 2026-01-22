import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def custom_optimizer(inputs, output, size_dict, memory_limit):
    return [(0, 1)] * (len(inputs) - 1)