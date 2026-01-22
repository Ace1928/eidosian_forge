from typing import Any, Callable, List, Optional, Union, cast
from ..language import (
from ..pyutils import Undefined
from ..type import (
from .type_from_ast import type_from_ast
def leave_enum_value(self) -> None:
    self._enum_value = None