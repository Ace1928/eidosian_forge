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
def resolve_default_value(self):
    """ Resolves a class name into a class so that it can be used to
            return the class as the default value of the trait.
        """
    if isinstance(self.klass, str):
        try:
            self.resolve_class(None, None, None)
            del self.validate
        except:
            raise TraitError('Could not resolve %s into a valid class' % self.klass)
    return self.klass