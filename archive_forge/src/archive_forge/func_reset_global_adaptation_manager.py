from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def reset_global_adaptation_manager():
    """ Set the global adaptation manager to a new AdaptationManager instance.
    """
    global adaptation_manager
    adaptation_manager = AdaptationManager()