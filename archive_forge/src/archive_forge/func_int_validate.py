import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
def int_validate(self, object, name, value):
    """ Validate that the value is an int value in the specified range.
        """
    original_value = value
    try:
        value = _validate_int(value)
    except TypeError:
        self.error(object, name, original_value)
    if (self._low is None or (self._exclude_low and self._low < value) or (not self._exclude_low and self._low <= value)) and (self._high is None or (self._exclude_high and self._high > value) or (not self._exclude_high and self._high >= value)):
        return value
    self.error(object, name, original_value)