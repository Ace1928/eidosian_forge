from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
def field_serializer(*fields: str, mode: Literal['plain', 'wrap']='plain', return_type: Any=PydanticUndefined, when_used: Literal['always', 'unless-none', 'json', 'json-unless-none']='always', check_fields: bool | None=None) -> Callable[[Any], Any]:
    """Decorator that enables custom field serialization.

    In the below example, a field of type `set` is used to mitigate duplication. A `field_serializer` is used to serialize the data as a sorted list.

    ```python
    from typing import Set

    from pydantic import BaseModel, field_serializer

    class StudentModel(BaseModel):
        name: str = 'Jane'
        courses: Set[str]

        @field_serializer('courses', when_used='json')
        def serialize_courses_in_order(courses: Set[str]):
            return sorted(courses)

    student = StudentModel(courses={'Math', 'Chemistry', 'English'})
    print(student.model_dump_json())
    #> {"name":"Jane","courses":["Chemistry","English","Math"]}
    ```

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

    Four signatures are supported:

    - `(self, value: Any, info: FieldSerializationInfo)`
    - `(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo)`
    - `(value: Any, info: SerializationInfo)`
    - `(value: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

    Args:
        fields: Which field(s) the method should be called on.
        mode: The serialization mode.

            - `plain` means the function will be called instead of the default serialization logic,
            - `wrap` means the function will be called with an argument to optionally call the
               default serialization logic.
        return_type: Optional return type for the function, if omitted it will be inferred from the type annotation.
        when_used: Determines the serializer will be used for serialization.
        check_fields: Whether to check that the fields actually exist on the model.

    Returns:
        The decorator function.
    """

    def dec(f: Callable[..., Any] | staticmethod[Any, Any] | classmethod[Any, Any, Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        dec_info = _decorators.FieldSerializerDecoratorInfo(fields=fields, mode=mode, return_type=return_type, when_used=when_used, check_fields=check_fields)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec