from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
 Returns all public methods on a class.

            Returns a dictionary containing all public methods keyed by name.
        