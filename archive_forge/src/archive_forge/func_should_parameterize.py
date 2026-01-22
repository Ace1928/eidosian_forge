import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
@staticmethod
def should_parameterize(op: 'cirq.Operation') -> bool:
    return isinstance(op.gate, (ops.PhasedFSimGate, ops.ISwapPowGate, ops.FSimGate))