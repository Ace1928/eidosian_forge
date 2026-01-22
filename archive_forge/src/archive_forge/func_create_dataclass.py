from __future__ import annotations as _annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, NoReturn, TypeVar, overload
from typing_extensions import Literal, TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .fields import Field, FieldInfo
def create_dataclass(cls: type[Any]) -> type[PydanticDataclass]:
    """Create a Pydantic dataclass from a regular dataclass.

        Args:
            cls: The class to create the Pydantic dataclass from.

        Returns:
            A Pydantic dataclass.
        """
    original_cls = cls
    config_dict = config
    if config_dict is None:
        cls_config = getattr(cls, '__pydantic_config__', None)
        if cls_config is not None:
            config_dict = cls_config
    config_wrapper = _config.ConfigWrapper(config_dict)
    decorators = _decorators.DecoratorInfos.build(cls)
    original_doc = cls.__doc__
    if _pydantic_dataclasses.is_builtin_dataclass(cls):
        original_doc = None
        bases = (cls,)
        if issubclass(cls, Generic):
            generic_base = Generic[cls.__parameters__]
            bases = bases + (generic_base,)
        cls = types.new_class(cls.__name__, bases)
    make_pydantic_fields_compatible(cls)
    cls = dataclasses.dataclass(cls, init=True, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen, **kwargs)
    cls.__pydantic_decorators__ = decorators
    cls.__doc__ = original_doc
    cls.__module__ = original_cls.__module__
    cls.__qualname__ = original_cls.__qualname__
    pydantic_complete = _pydantic_dataclasses.complete_dataclass(cls, config_wrapper, raise_errors=False, types_namespace=None)
    cls.__pydantic_complete__ = pydantic_complete
    return cls