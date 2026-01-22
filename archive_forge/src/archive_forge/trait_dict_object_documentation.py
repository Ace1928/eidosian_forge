import copy
import sys
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import Undefined, _validate_everything
from traits.trait_errors import TraitError
 Update self with the contents of other.

            Parameters
            ----------
            other : mapping or iterable of (key, value) pairs
                Values to be added to this dictionary.
            