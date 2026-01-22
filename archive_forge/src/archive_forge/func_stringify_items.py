from __future__ import annotations
from typing import Any, List, Tuple, Union, Mapping, TypeVar
from urllib.parse import parse_qs, urlencode
from typing_extensions import Literal, get_args
from ._types import NOT_GIVEN, NotGiven, NotGivenOr
from ._utils import flatten
def stringify_items(self, params: Params, *, array_format: NotGivenOr[ArrayFormat]=NOT_GIVEN, nested_format: NotGivenOr[NestedFormat]=NOT_GIVEN) -> list[tuple[str, str]]:
    opts = Options(qs=self, array_format=array_format, nested_format=nested_format)
    return flatten([self._stringify_item(key, value, opts) for key, value in params.items()])