from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def definition_reference_schema(schema_ref: str, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> DefinitionReferenceSchema:
    """
    Returns a schema that points to a schema stored in "definitions", this is useful for nested recursive
    models and also when you want to define validators separately from the main schema, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    schema_definition = core_schema.definition_reference_schema('list-schema')
    schema = core_schema.definitions_schema(
        schema=schema_definition,
        definitions=[
            core_schema.list_schema(items_schema=schema_definition, ref='list-schema'),
        ],
    )
    v = SchemaValidator(schema)
    assert v.validate_python([()]) == [[]]
    ```

    Args:
        schema_ref: The schema ref to use for the definition reference schema
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    return _dict_not_none(type='definition-ref', schema_ref=schema_ref, ref=ref, metadata=metadata, serialization=serialization)