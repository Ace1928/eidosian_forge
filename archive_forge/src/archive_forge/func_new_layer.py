import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def new_layer(self, previous_single_qubit_layer: 'cirq.Moment') -> 'cirq.Moment':
    return circuits.Moment((v.on(q) for q, v in self.fixed_single_qubit_layer.items()))