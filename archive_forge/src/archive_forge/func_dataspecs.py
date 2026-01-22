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
@classmethod
@lru_cache(None)
def dataspecs(cls) -> dict[str, DataSpec]:
    """ Collect the names of all ``DataSpec`` properties on this class.

        This method *always* traverses the class hierarchy and includes
        properties defined on any parent classes.

        Returns:
            set[str] : names of ``DataSpec`` properties

        """
    from .property.dataspec import DataSpec
    return {k: v for k, v in cls.properties(_with_props=True).items() if isinstance(v, DataSpec)}