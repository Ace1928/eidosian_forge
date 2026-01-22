import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def circuit_from_schedule_from_protos(ops) -> cirq.Circuit:
    """Convert protos into a Circuit."""
    result = []
    for op in ops:
        xmon_op = xmon_op_from_proto(op)
        result.append(xmon_op)
    ret = cirq.Circuit(result)
    return ret