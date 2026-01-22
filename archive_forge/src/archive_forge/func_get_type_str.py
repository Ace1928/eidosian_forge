from enum import Enum
from functools import lru_cache
from typing import (
@lru_cache
def get_type_str(cls: Union[type, str, None]) -> str:
    """Return a string representing the type ``cls``.

    If cls is a built-in type, such as 'str', returns the unqualified
        name.

    If cls is a parametrized generic such as List[str], or a special typing
        form such as Optional[int], returns the string representation of cls.

    Otherwise, returns the fully-qualified class name, including the module.
    """
    if cls is None or cls is type(None):
        return 'None'
    if isinstance(cls, str):
        return cls
    if isinstance(cls, ForwardRef):
        return cls.__forward_arg__
    if isinstance(cls, _SpecialForm):
        return cls._name
    orig_type = get_origin(cls)
    if orig_type is not None:
        orig_args = get_args(cls)
        if orig_args:
            return f'{get_type_str(orig_type)}[{', '.join((get_type_str(arg) for arg in orig_args))}]'
        return get_type_str(orig_type)
    if getattr(cls, '__module__', None) in ('builtins', None):
        return cls.__name__
    return f'{cls.__module__}.{cls.__qualname__}'