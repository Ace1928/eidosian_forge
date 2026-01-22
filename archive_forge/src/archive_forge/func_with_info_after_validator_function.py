from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def with_info_after_validator_function(function: WithInfoValidatorFunction, schema: CoreSchema, *, field_name: str | None=None, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> AfterValidatorFunctionSchema:
    """
    Returns a schema that calls a validator function after validation, the function is called with
    an `info` argument, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    def fn(v: str, info: core_schema.ValidationInfo) -> str:
        assert info.data is not None
        assert info.field_name is not None
        return v + 'world'

    func_schema = core_schema.with_info_after_validator_function(
        function=fn, schema=core_schema.str_schema(), field_name='a'
    )
    schema = core_schema.typed_dict_schema({'a': core_schema.typed_dict_field(func_schema)})

    v = SchemaValidator(schema)
    assert v.validate_python({'a': b'hello '}) == {'a': 'hello world'}
    ```

    Args:
        function: The validator function to call after the schema is validated
        schema: The schema to validate before the validator function
        field_name: The name of the field this validators is applied to, if any
        ref: optional unique identifier of the schema, used to reference the schema in other places
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    return _dict_not_none(type='function-after', function=_dict_not_none(type='with-info', function=function, field_name=field_name), schema=schema, ref=ref, metadata=metadata, serialization=serialization)