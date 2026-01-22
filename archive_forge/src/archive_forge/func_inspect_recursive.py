from inspect import (
from typing import Any, List
from .undefined import Undefined
def inspect_recursive(value: Any, seen_values: List) -> str:
    if value is None or value is Undefined or isinstance(value, (bool, float, complex)):
        return repr(value)
    if isinstance(value, (int, str, bytes, bytearray)):
        return trunc_str(repr(value))
    if len(seen_values) < max_recursive_depth and value not in seen_values:
        inspect_method = getattr(value, '__inspect__', None)
        if inspect_method is not None and callable(inspect_method):
            s = inspect_method()
            if isinstance(s, str):
                return trunc_str(s)
            seen_values = [*seen_values, value]
            return inspect_recursive(s, seen_values)
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            if not value:
                return repr(value)
            seen_values = [*seen_values, value]
            if isinstance(value, list):
                items = value
            elif isinstance(value, dict):
                items = list(value.items())
            else:
                items = list(value)
            items = trunc_list(items)
            if isinstance(value, dict):
                s = ', '.join(('...' if v is ELLIPSIS else inspect_recursive(v[0], seen_values) + ': ' + inspect_recursive(v[1], seen_values) for v in items))
            else:
                s = ', '.join(('...' if v is ELLIPSIS else inspect_recursive(v, seen_values) for v in items))
            if isinstance(value, tuple):
                if len(items) == 1:
                    return f'({s},)'
                return f'({s})'
            if isinstance(value, (dict, set)):
                return '{' + s + '}'
            if isinstance(value, frozenset):
                return f'frozenset({{{s}}})'
            return f'[{s}]'
    elif isinstance(value, (list, tuple, dict, set, frozenset)):
        if not value:
            return repr(value)
        if isinstance(value, list):
            return '[...]'
        if isinstance(value, tuple):
            return '(...)'
        if isinstance(value, dict):
            return '{...}'
        if isinstance(value, set):
            return 'set(...)'
        return 'frozenset(...)'
    if isinstance(value, Exception):
        type_ = 'exception'
        value = type(value)
    elif isclass(value):
        type_ = 'exception class' if issubclass(value, Exception) else 'class'
    elif ismethod(value):
        type_ = 'method'
    elif iscoroutinefunction(value):
        type_ = 'coroutine function'
    elif isasyncgenfunction(value):
        type_ = 'async generator function'
    elif isgeneratorfunction(value):
        type_ = 'generator function'
    elif isfunction(value):
        type_ = 'function'
    elif iscoroutine(value):
        type_ = 'coroutine'
    elif isasyncgen(value):
        type_ = 'async generator'
    elif isgenerator(value):
        type_ = 'generator'
    else:
        from ..type import GraphQLDirective, GraphQLNamedType, GraphQLScalarType, GraphQLWrappingType
        if isinstance(value, (GraphQLDirective, GraphQLNamedType, GraphQLScalarType, GraphQLWrappingType)):
            return str(value)
        try:
            name = type(value).__name__
            if not name or '<' in name or '>' in name:
                raise AttributeError
        except AttributeError:
            return '<object>'
        else:
            return f'<{name} instance>'
    try:
        name = value.__name__
        if not name or '<' in name or '>' in name:
            raise AttributeError
    except AttributeError:
        return f'<{type_}>'
    else:
        return f'<{type_} {name}>'