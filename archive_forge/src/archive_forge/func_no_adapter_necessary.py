from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def no_adapter_necessary(adaptee):
    """ An adapter factory used to register that a protocol provides another.

    See 'register_provides' for details.

    """
    return adaptee