from heapq import heappop, heappush
import inspect
import itertools
import functools
from traits.adaptation.adaptation_error import AdaptationError
from traits.has_traits import HasTraits
from traits.trait_types import Dict, List, Str
def register_factory(self, factory, from_protocol, to_protocol):
    """ Register an adapter factory.

        This is a simply a convenience method that creates and registers an
        'AdaptationOffer' from the given arguments.

        """
    from traits.adaptation.adaptation_offer import AdaptationOffer
    self.register_offer(AdaptationOffer(factory=factory, from_protocol=from_protocol, to_protocol=to_protocol))