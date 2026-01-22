import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def has_placeholder(self, alias_id: Hashable) -> bool:
    return alias_id in self._alias_id_to_placeholder