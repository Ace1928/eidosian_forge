from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
def deepfreeze(o, custom_converters=None, custom_inverse_converters=None):
    """
    Converts the object and all the objects nested in it in its
    immutable counterparts.
    
    The conversion map is in getFreezeConversionMap().
    
    You can register a new conversion using `register()` You can also
    pass a map of custom converters with `custom_converters` and a map
    of custom inverse converters with `custom_inverse_converters`,
    without using `register()`.
    
    By default, if the type is not registered and has a `__dict__`
    attribute, it's converted to the `frozendict` of that `__dict__`.
    
    This function assumes that hashable == immutable (that is not
    always true).
    
    This function uses recursion, with all the limits of recursions in
    Python.
    
    Where is a good old tail call when you need it?
    """
    from frozendict import frozendict
    if custom_converters is None:
        custom_converters = frozendict()
    if custom_inverse_converters is None:
        custom_inverse_converters = frozendict()
    for type_i, converter in custom_converters.items():
        if not issubclass(type(type_i), type):
            raise ValueError(f'{type_i} in `custom_converters` parameter is not a ' + 'type')
        try:
            converter.__call__
        except AttributeError:
            raise ValueError(f'converter for {type_i} in `custom_converters` ' + 'parameter is not a callable')
    for type_i, converter in custom_inverse_converters.items():
        if not issubclass(type(type_i), type):
            raise ValueError(f'{type_i} in `custom_inverse_converters` parameter ' + 'is not a type')
        try:
            converter.__call__
        except AttributeError:
            raise ValueError(f'converter for {type_i} in  ' + '`custom_inverse_converters`parameter is not a callable')
    type_o = type(o)
    freeze_types = tuple(custom_converters.keys()) + getFreezeTypes()
    base_type_o = None
    for freeze_type in freeze_types:
        if isinstance(o, freeze_type):
            base_type_o = freeze_type
            break
    if base_type_o is None:
        try:
            o.__dict__
        except AttributeError:
            pass
        else:
            return frozendict(o.__dict__)
        try:
            hash(o)
        except TypeError:
            pass
        else:
            return o
        supported_types = ', '.join((x.__name__ for x in freeze_types))
        err = f'type {type_o} is not hashable or is not equal or a ' + f'subclass of the supported types: {supported_types}'
        raise TypeError(err)
    freeze_conversion_map = getFreezeConversionMap()
    freeze_conversion_map = freeze_conversion_map | custom_converters
    if base_type_o in _freeze_types_plain:
        return freeze_conversion_map[base_type_o](o)
    if not isIterableNotString(o):
        return freeze_conversion_map[base_type_o](o)
    freeze_conversion_inverse_map = getFreezeConversionInverseMap()
    freeze_conversion_inverse_map = freeze_conversion_inverse_map | custom_inverse_converters
    frozen_type = base_type_o in freeze_conversion_inverse_map
    if frozen_type:
        o = freeze_conversion_inverse_map[base_type_o](o)
    from copy import copy
    o_copy = copy(o)
    for k, v in getItems(o_copy)(o_copy):
        o_copy[k] = deepfreeze(v, custom_converters=custom_converters, custom_inverse_converters=custom_inverse_converters)
    try:
        freeze = freeze_conversion_map[base_type_o]
    except KeyError:
        if frozen_type:
            freeze = type_o
        else:
            raise
    return freeze(o_copy)