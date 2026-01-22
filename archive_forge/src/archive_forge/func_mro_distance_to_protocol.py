from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
@staticmethod
def mro_distance_to_protocol(from_type, to_protocol):
    """ Return the distance in the MRO from 'from_type' to 'to_protocol'.

        If `from_type` provides `to_protocol`, returns the distance between
        `from_type` and the super-most class in the MRO hierarchy providing
        `to_protocol` (that's where the protocol was provided in the first
        place).

        If `from_type` does not provide `to_protocol`, return None.

        """
    if not AdaptationManager.provides_protocol(from_type, to_protocol):
        return None
    supertypes = inspect.getmro(from_type)[1:]
    distance = 0
    for t in supertypes:
        if AdaptationManager.provides_protocol(t, to_protocol):
            distance += 1
        else:
            break
    return distance