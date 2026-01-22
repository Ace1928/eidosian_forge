from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_union(type_: Type) -> bool:
    return is_generic(type_) and type_.__origin__ == Union