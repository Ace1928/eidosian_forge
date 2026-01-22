from typing import Any, Callable, List, Optional, Union, cast
from ..language import (
from ..pyutils import Undefined
from ..type import (
from .type_from_ast import type_from_ast
def leave_argument(self) -> None:
    self._argument = None
    del self._default_value_stack[-1:]
    del self._input_type_stack[-1:]