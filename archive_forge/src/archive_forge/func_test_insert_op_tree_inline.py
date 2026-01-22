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
def test_insert_op_tree_inline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit([cirq.Moment([cirq.H(a)])])
    op_tree_list = [(1, 1, [cirq.H(a), cirq.X(b)], [a, b]), (0, 0, [cirq.X(b)], [b]), (4, 3, [cirq.H(b)], [b]), (5, 3, [cirq.H(a)], [a]), (-2, 0, [cirq.X(b)], [b]), (-5, 0, [cirq.CZ(a, b)], [a])]
    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]