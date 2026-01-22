import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
@property
def flat_inputs(self) -> List[trace.TraceType]:
    """Flat tensor inputs accepted by this FunctionType."""
    if not hasattr(self, '_cached_flat_inputs'):
        cached_flat_inputs = []
        for p in self.parameters.values():
            cached_flat_inputs.extend(p.type_constraint._flatten())
        self._cached_flat_inputs = cached_flat_inputs
    return self._cached_flat_inputs