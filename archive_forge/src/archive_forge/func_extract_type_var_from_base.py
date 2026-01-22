from __future__ import annotations
from typing import Any, TypeVar, Iterable, cast
from collections import abc as _c_abc
from typing_extensions import Required, Annotated, get_args, get_origin
from .._types import InheritsGeneric
from .._compat import is_union as _is_union
def extract_type_var_from_base(typ: type, *, generic_bases: tuple[type, ...], index: int, failure_message: str | None=None) -> type:
    """Given a type like `Foo[T]`, returns the generic type variable `T`.

    This also handles the case where a concrete subclass is given, e.g.
    ```py
    class MyResponse(Foo[bytes]):
        ...

    extract_type_var(MyResponse, bases=(Foo,), index=0) -> bytes
    ```

    And where a generic subclass is given:
    ```py
    _T = TypeVar('_T')
    class MyResponse(Foo[_T]):
        ...

    extract_type_var(MyResponse[bytes], bases=(Foo,), index=0) -> bytes
    ```
    """
    cls = cast(object, get_origin(typ) or typ)
    if cls in generic_bases:
        return extract_type_arg(typ, index)
    if isinstance(cls, InheritsGeneric):
        target_base_class: Any | None = None
        for base in cls.__orig_bases__:
            if base.__origin__ in generic_bases:
                target_base_class = base
                break
        if target_base_class is None:
            raise RuntimeError(f'Could not find the generic base class;\nThis should never happen;\nDoes {cls} inherit from one of {generic_bases} ?')
        extracted = extract_type_arg(target_base_class, index)
        if is_typevar(extracted):
            return extract_type_arg(typ, index)
        return extracted
    raise RuntimeError(failure_message or f'Could not resolve inner type variable at index {index} for {typ}')