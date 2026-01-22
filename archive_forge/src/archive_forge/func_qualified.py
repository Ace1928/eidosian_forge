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
def qualified():
    module = cls.__view_module__
    model = cls.__view_model__
    if issubclass(cls, NonQualified):
        return model
    if not issubclass(cls, Qualified):
        head = module.split('.')[0]
        if head == 'bokeh' or head == '__main__' or '__implementation__' in cls.__dict__:
            return model
    return f'{module}.{model}'