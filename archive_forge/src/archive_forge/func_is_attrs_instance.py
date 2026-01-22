import warnings
from types import MappingProxyType
from typing import Any
from itemadapter._imports import attr, pydantic
def is_attrs_instance(obj: Any) -> bool:
    warnings.warn('itemadapter.utils.is_attrs_instance is deprecated and it will be removed in a future version', category=DeprecationWarning, stacklevel=2)
    from itemadapter.adapter import AttrsAdapter
    return AttrsAdapter.is_item(obj)