import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def random_gate(qubit: 'cirq.Qid') -> 'cirq.Gate':
    excluded_op = previous_single_qubit_layer.operation_at(qubit)
    excluded_gate = excluded_op.gate if excluded_op is not None else None
    g = self.single_qubit_gates[self.prng.randint(0, len(self.single_qubit_gates))]
    while g is excluded_gate:
        g = self.single_qubit_gates[self.prng.randint(0, len(self.single_qubit_gates))]
    return g