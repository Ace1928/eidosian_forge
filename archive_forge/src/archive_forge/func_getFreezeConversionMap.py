from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
def getFreezeConversionMap():
    return _freeze_conversion_map | _freeze_conversion_map_custom