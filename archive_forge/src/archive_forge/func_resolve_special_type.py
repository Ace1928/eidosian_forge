from enum import Enum
from functools import lru_cache
from typing import (
def resolve_special_type(type_: Any) -> Optional[Tuple[type, List[type]]]:
    """Converts special typing forms (Union[...], Optional[...]), and parametrized
    generics (List[...], Dict[...]) into a 2-tuple of its base type and arguments.
    If ``type_`` is a regular type, or an object, this function will return
    ``None``.

    Note that this function will only perform one level of recursion - the
    arguments of nested types will not be resolved:

        >>> resolve_special_type(List[List[int]])
        (<class 'list'>, [<class 'list'>])

    Further examples:
        >>> resolve_special_type(Union[str, int])
        (typing.Union, [<class 'str'>, <class 'int'>])
        >>> resolve_special_type(List[int])
        (<class 'list'>, [<class 'int'>])
        >>> resolve_special_type(List)
        (<class 'list'>, [])
        >>> resolve_special_type(list)
        None
    """
    orig_type = get_origin(type_)
    if orig_type is None:
        return None
    args = list(get_args(type_))
    type_ = orig_type
    for i, arg in enumerate(args):
        orig_type = get_origin(arg)
        if orig_type:
            args[i] = orig_type
    return (type_, args)