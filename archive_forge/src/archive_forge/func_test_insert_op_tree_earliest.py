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
def test_insert_op_tree_earliest():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit([cirq.Moment([cirq.H(a)])])
    op_tree_list = [(5, [1, 0], [cirq.X(a), cirq.X(b)], [a, b]), (1, [1], [cirq.H(b)], [b]), (-4, [0], [cirq.X(b)], [b])]
    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.EARLIEST)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index[i]) == op_list[i]