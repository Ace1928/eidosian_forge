from typing import (
import numpy as np
from cirq import protocols, _compat
from cirq.circuits import AbstractCircuit, Alignment, Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType
def concat_ragged(*circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str]=Alignment.LEFT) -> 'cirq.FrozenCircuit':
    return AbstractCircuit.concat_ragged(*circuits, align=align).freeze()