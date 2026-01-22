from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def export_constant(self, module_name: str, name: str) -> None:
    """Store export information for constants/string literals.

    Export information is stored in the module where constants/string literals
    are defined.

    e.g.
    ```python
    foo = 1
    bar = 2
    tf_export("consts.foo").export_constant(__name__, 'foo')
    tf_export("consts.bar").export_constant(__name__, 'bar')
    ```

    Args:
      module_name: (string) Name of the module to store constant at.
      name: (string) Current constant name.
    """
    module = sys.modules[module_name]
    api_constants_attr = API_ATTRS[self._api_name].constants
    api_constants_attr_v1 = API_ATTRS_V1[self._api_name].constants
    if not hasattr(module, api_constants_attr):
        setattr(module, api_constants_attr, [])
    getattr(module, api_constants_attr).append((self._names, name))
    if not hasattr(module, api_constants_attr_v1):
        setattr(module, api_constants_attr_v1, [])
    getattr(module, api_constants_attr_v1).append((self._names_v1, name))