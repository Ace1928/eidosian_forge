from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
def val_func(v: Any) -> Any:
    if not annotation.func(v):
        raise PydanticCustomError('predicate_failed', f'Predicate {predicate_name}failed')
    return v