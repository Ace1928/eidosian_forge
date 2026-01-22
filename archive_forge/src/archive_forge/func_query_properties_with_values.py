from __future__ import annotations
import logging # isort:skip
import difflib
from typing import (
from weakref import WeakSet
from ..settings import settings
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import (
from .types import ID
def query_properties_with_values(self, query: Callable[[PropertyDescriptor[Any]], bool], *, include_defaults: bool=True, include_undefined: bool=False) -> dict[str, Any]:
    """ Query the properties values of |HasProps| instances with a
        predicate.

        Args:
            query (callable) :
                A callable that accepts property descriptors and returns True
                or False

            include_defaults (bool, optional) :
                Whether to include properties that have not been explicitly
                set by a user (default: True)

        Returns:
            dict : mapping of property names and values for matching properties

        """
    themed_keys: set[str] = set()
    result: dict[str, Any] = {}
    keys = self.properties(_with_props=True)
    if include_defaults:
        selected_keys = set(keys)
    else:
        selected_keys = set(self._property_values.keys()) | set(self._unstable_default_values.keys())
        themed_values = self.themed_values()
        if themed_values is not None:
            themed_keys = set(themed_values.keys())
            selected_keys |= themed_keys
    for key in keys:
        descriptor = self.lookup(key)
        if not query(descriptor):
            continue
        try:
            value = descriptor.get_value(self)
        except UnsetValueError:
            if include_undefined:
                value = Undefined
            else:
                raise
        else:
            if key not in selected_keys:
                continue
            if not include_defaults and key not in themed_keys:
                if isinstance(value, PropertyValueContainer) and key in self._unstable_default_values:
                    continue
        result[key] = value
    return result