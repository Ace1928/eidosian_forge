import copy
import copyreg
from itertools import chain
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
 Overridden to make sure we call our custom __getstate__.
        