import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def update_naming_scope(self, naming_scope: Optional[str]) -> None:
    self._naming_scope = naming_scope