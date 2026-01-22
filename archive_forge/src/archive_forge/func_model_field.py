from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def model_field(schema: CoreSchema, *, validation_alias: str | list[str | int] | list[list[str | int]] | None=None, serialization_alias: str | None=None, serialization_exclude: bool | None=None, frozen: bool | None=None, metadata: Any=None) -> ModelField:
    """
    Returns a schema for a model field, e.g.:

    ```py
    from pydantic_core import core_schema

    field = core_schema.model_field(schema=core_schema.int_schema())
    ```

    Args:
        schema: The schema to use for the field
        validation_alias: The alias(es) to use to find the field in the validation data
        serialization_alias: The alias to use as a key when serializing
        serialization_exclude: Whether to exclude the field when serializing
        frozen: Whether the field is frozen
        metadata: Any other information you want to include with the schema, not used by pydantic-core
    """
    return _dict_not_none(type='model-field', schema=schema, validation_alias=validation_alias, serialization_alias=serialization_alias, serialization_exclude=serialization_exclude, frozen=frozen, metadata=metadata)