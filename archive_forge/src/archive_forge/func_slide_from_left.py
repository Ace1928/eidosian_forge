import re
from collections import OrderedDict
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit, ControlFlowOp
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, PowerModifier
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from ..exceptions import VisualizationError
def slide_from_left(self, node, index):
    """Insert node into first layer where there is no conflict going l > r"""
    measure_layer = None
    if isinstance(node.op, Measure):
        measure_bit = next((bit for bit in self.measure_map if node.cargs[0] == bit))
    if not self:
        inserted = True
        self.append([node])
    else:
        inserted = False
        curr_index = index
        last_insertable_index = -1
        index_stop = -1
        if (condition := getattr(node.op, 'condition', None)) is not None:
            index_stop = max((self.measure_map[bit] for bit in condition_resources(condition).clbits), default=index_stop)
        if node.cargs:
            for carg in node.cargs:
                try:
                    carg_bit = next((bit for bit in self.measure_map if carg == bit))
                    if self.measure_map[carg_bit] > index_stop:
                        index_stop = self.measure_map[carg_bit]
                except StopIteration:
                    pass
        while curr_index > index_stop:
            if self.is_found_in(node, self[curr_index]):
                break
            if self.insertable(node, self[curr_index]):
                last_insertable_index = curr_index
            curr_index = curr_index - 1
        if last_insertable_index >= 0:
            inserted = True
            self[last_insertable_index].append(node)
            measure_layer = last_insertable_index
        else:
            inserted = False
            curr_index = index
            while curr_index < len(self):
                if self.insertable(node, self[curr_index]):
                    self[curr_index].append(node)
                    measure_layer = curr_index
                    inserted = True
                    break
                curr_index = curr_index + 1
    if not inserted:
        self.append([node])
    if isinstance(node.op, Measure):
        if not measure_layer:
            measure_layer = len(self) - 1
        if measure_layer > self.measure_map[measure_bit]:
            self.measure_map[measure_bit] = measure_layer