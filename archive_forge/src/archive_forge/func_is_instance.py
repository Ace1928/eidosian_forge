from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_instance(value: Any, type_: Type) -> bool:
    if type_ == Any:
        return True
    elif is_union(type_):
        return any((is_instance(value, t) for t in extract_generic(type_)))
    elif is_generic_collection(type_):
        origin = extract_origin_collection(type_)
        if not isinstance(value, origin):
            return False
        if not extract_generic(type_):
            return True
        if isinstance(value, tuple):
            tuple_types = extract_generic(type_)
            if len(tuple_types) == 1 and tuple_types[0] == ():
                return len(value) == 0
            elif len(tuple_types) == 2 and tuple_types[1] is ...:
                return all((is_instance(item, tuple_types[0]) for item in value))
            else:
                if len(tuple_types) != len(value):
                    return False
                return all((is_instance(item, item_type) for item, item_type in zip(value, tuple_types)))
        if isinstance(value, Mapping):
            key_type, val_type = extract_generic(type_, defaults=(Any, Any))
            for key, val in value.items():
                if not is_instance(key, key_type) or not is_instance(val, val_type):
                    return False
            return True
        return all((is_instance(item, extract_generic(type_, defaults=(Any,))[0]) for item in value))
    elif is_new_type(type_):
        return is_instance(value, extract_new_type(type_))
    elif is_literal(type_):
        return value in extract_generic(type_)
    elif is_init_var(type_):
        return is_instance(value, extract_init_var(type_))
    elif is_type_generic(type_):
        return is_subclass(value, extract_generic(type_)[0])
    else:
        try:
            if isinstance(value, (int, float)) and type_ in [float, complex]:
                return True
            return isinstance(value, type_)
        except TypeError:
            return False