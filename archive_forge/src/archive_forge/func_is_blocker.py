import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def is_blocker(op):
    if op.gate.label == 'F':
        return False
    if op.gate.label == 'T':
        return True
    return prng.rand() < 0.5