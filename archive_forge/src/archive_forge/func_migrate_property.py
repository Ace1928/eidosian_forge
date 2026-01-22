import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
def migrate_property(name, property, property_info, class_dict):
    """ Migrates an existing property to the class being defined
    (allowing for method overrides).
    """
    get = _property_method(class_dict, '_get_' + name)
    set = _property_method(class_dict, '_set_' + name)
    val = _property_method(class_dict, '_validate_' + name)
    if get is not None or set is not None or val is not None:
        old_get, old_set, old_val = property_info
        return Property(get or old_get, set or old_set, val or old_val, True, **property.__dict__)
    return property