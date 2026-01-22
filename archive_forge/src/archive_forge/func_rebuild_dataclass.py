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
def rebuild_dataclass(cls: type[PydanticDataclass], *, force: bool=False, raise_errors: bool=True, _parent_namespace_depth: int=2, _types_namespace: dict[str, Any] | None=None) -> bool | None:
    """Try to rebuild the pydantic-core schema for the dataclass.

    This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
    the initial attempt to build the schema, and automatic rebuilding fails.

    This is analogous to `BaseModel.model_rebuild`.

    Args:
        cls: The class to rebuild the pydantic-core schema for.
        force: Whether to force the rebuilding of the schema, defaults to `False`.
        raise_errors: Whether to raise errors, defaults to `True`.
        _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
        _types_namespace: The types namespace, defaults to `None`.

    Returns:
        Returns `None` if the schema is already "complete" and rebuilding was not required.
        If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
    """
    if not force and cls.__pydantic_complete__:
        return None
    else:
        if _types_namespace is not None:
            types_namespace: dict[str, Any] | None = _types_namespace.copy()
        else:
            if _parent_namespace_depth > 0:
                frame_parent_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth) or {}
                types_namespace = frame_parent_ns
            else:
                types_namespace = {}
            types_namespace = _typing_extra.get_cls_types_namespace(cls, types_namespace)
        return _pydantic_dataclasses.complete_dataclass(cls, _config.ConfigWrapper(cls.__pydantic_config__, check=False), raise_errors=raise_errors, types_namespace=types_namespace)