from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def with_default_schema(schema: CoreSchema, *, default: Any=PydanticUndefined, default_factory: Callable[[], Any] | None=None, on_error: Literal['raise', 'omit', 'default'] | None=None, validate_default: bool | None=None, strict: bool | None=None, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> WithDefaultSchema:
    """
    Returns a schema that adds a default value to the given schema, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    schema = core_schema.with_default_schema(core_schema.str_schema(), default='hello')
    wrapper_schema = core_schema.typed_dict_schema(
        {'a': core_schema.typed_dict_field(schema)}
    )
    v = SchemaValidator(wrapper_schema)
    assert v.validate_python({}) == v.validate_python({'a': 'hello'})
    ```

    Args:
        schema: The schema to add a default value to
        default: The default value to use
        default_factory: A function that returns the default value to use
        on_error: What to do if the schema validation fails. One of 'raise', 'omit', 'default'
        validate_default: Whether the default value should be validated
        strict: Whether the underlying schema should be validated with strict mode
        ref: optional unique identifier of the schema, used to reference the schema in other places
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    s = _dict_not_none(type='default', schema=schema, default_factory=default_factory, on_error=on_error, validate_default=validate_default, strict=strict, ref=ref, metadata=metadata, serialization=serialization)
    if default is not PydanticUndefined:
        s['default'] = default
    return s