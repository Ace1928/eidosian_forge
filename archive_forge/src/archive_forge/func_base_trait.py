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
def base_trait(self, name):
    """Returns the base trait definition for a trait attribute.

        This method is similar to the trait() method, and returns a
        different result only in the case where the trait attribute defined by
        *name* is a delegate. In this case, the base_trait() method follows the
        delegation chain until a non-delegated trait attribute is reached, and
        returns the definition of that attribute's trait as the result.

        Parameters
        ----------
        name : str
            Name of the attribute whose trait definition is returned.
        """
    return self._trait(name, -2)