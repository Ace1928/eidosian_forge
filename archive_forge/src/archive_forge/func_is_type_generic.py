from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_type_generic(type_: Type) -> bool:
    try:
        return type_.__origin__ in (type, Type)
    except AttributeError:
        return False