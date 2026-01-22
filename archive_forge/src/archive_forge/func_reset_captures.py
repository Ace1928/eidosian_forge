import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def reset_captures(self, tensors, placeholders):
    """Set the captures with the provided list of captures & placeholder."""
    self._by_val_external = MutationAwareDict()
    self._by_val_internal = MutationAwareDict()
    self._by_val_tracetype = MutationAwareDict()
    for external, internal in zip(tensors, placeholders):
        key = id(external)
        self._by_val_external[key] = external
        self._by_val_internal[key] = internal
        self._by_val_tracetype[key] = trace_type.from_value(external)