import itertools
import numpy as np
import pytest
import cirq
import sympy
def random_unitary(seed):
    return cirq.testing.random_unitary(4, random_state=seed)