import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def col_subscripts(qs: Sequence['cirq.Qid']) -> str:
    return ''.join((qubit_to_col_subscript[q] for q in qs))