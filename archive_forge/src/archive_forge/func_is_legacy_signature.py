import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
@property
def is_legacy_signature(self) -> bool:
    """If the value is from a legacy signature representation.

    Legacy signature representations include tf.function.input_signature and
    ConcreteFunction.structured_input_signature.
    """
    return self._is_legacy_signature