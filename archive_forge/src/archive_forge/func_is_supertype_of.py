import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def is_supertype_of(self, other: 'FunctionType') -> bool:
    """Returns True if self is a supertype of other FunctionType."""
    if len(self.parameters) != len(other.parameters):
        return False
    for self_param, other_param in zip(self.parameters.values(), other.parameters.values()):
        if not self_param.is_subtype_of(other_param):
            return False
    if not all((name in other.captures for name in self.captures)):
        return False
    return all((capture_type.is_subtype_of(other.captures[name]) for name, capture_type in self.captures.items()))