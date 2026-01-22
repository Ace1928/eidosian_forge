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
@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(_cls: type[_T], *, init: Literal[False]=False, repr: bool=True, eq: bool=True, order: bool=False, unsafe_hash: bool=False, frozen: bool=False, config: ConfigDict | type[object] | None=None, validate_on_init: bool | None=None) -> type[PydanticDataclass]:
    ...