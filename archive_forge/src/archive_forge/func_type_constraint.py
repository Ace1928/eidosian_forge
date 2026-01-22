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
def type_constraint(self) -> Optional[trace.TraceType]:
    """A supertype that the parameter's type must subtype for validity."""
    return self.annotation if self.annotation is not self.empty else None