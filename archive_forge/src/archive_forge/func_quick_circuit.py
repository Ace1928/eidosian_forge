from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def quick_circuit(*moments: Iterable[cirq.OP_TREE]) -> cirq.Circuit:
    return cirq.Circuit([cirq.Moment(cast(Iterable[cirq.Operation], cirq.flatten_op_tree(m))) for m in moments])