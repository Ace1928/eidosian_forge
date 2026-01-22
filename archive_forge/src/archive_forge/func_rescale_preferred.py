import copy
from functools import wraps
import numpy as np
from . import markup
from .dimensionality import Dimensionality, p_dict
from .registry import unit_registry
from .decorators import with_doc
def rescale_preferred(self):
    """
        Return a copy of the quantity converted to the preferred units and scale.
        These will be identified from among the compatible units specified in the
        list PREFERRED in this module. For example, a voltage quantity might be
        converted to `mV`:
        ```
        import quantities as pq
        pq.quantity.PREFERRED = [pq.mV, pq.pA]
        old = 3.1415 * pq.V
        new = old.rescale_preferred() # `new` will be 3141.5 mV.
        ```
        """
    units_str = str(self.simplified.dimensionality)
    for preferred in PREFERRED:
        if units_str == str(preferred.simplified.dimensionality):
            return self.rescale(preferred)
    raise Exception("Preferred units for '%s' (or equivalent) not specified in quantites.quantity.PREFERRED." % self.dimensionality)