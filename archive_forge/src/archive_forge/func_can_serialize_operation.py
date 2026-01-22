from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import numbers
import abc
import numpy as np
import cirq
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto
def can_serialize_operation(self, op: cirq.Operation) -> bool:
    """Whether the given operation can be serialized by this serializer."""
    return self.can_serialize_predicate(op)