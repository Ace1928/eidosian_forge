from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def parse_value(self, value):
    if isinstance(value, Hashable):
        enum_value = self._name_lookup.get(value)
        if enum_value:
            return enum_value.value
    return None