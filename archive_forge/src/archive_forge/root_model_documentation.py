from __future__ import annotations as _annotations
import typing
from copy import copy, deepcopy
from pydantic_core import PydanticUndefined
from . import PydanticUserError
from ._internal import _model_construction, _repr
from .main import BaseModel, _object_setattr
This method is included just to get a more accurate return type for type checkers.
            It is included in this `if TYPE_CHECKING:` block since no override is actually necessary.

            See the documentation of `BaseModel.model_dump` for more details about the arguments.
            