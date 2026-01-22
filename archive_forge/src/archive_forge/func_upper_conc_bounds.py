import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def upper_conc_bounds(self, init_concs, min_=min, dtype=None, skip_keys=(0,)):
    """Calculates upper concentration bounds per substance based on substance composition.

        Parameters
        ----------
        init_concs : dict or array_like
            Per substance initial conidtions.
        min_ : callbable
        dtype : dtype or None
        skip_keys : tuple
            What composition keys to skip.

        Returns
        -------
        numpy.ndarray :
            Per substance upper limit (ordered as :attr:`substances`).

        Notes
        -----
        The function does not take into account whether there actually exists a
        reaction path leading to a substance. Note also that the upper limit is
        per substance, i.e. the sum of all upper bounds amount to more substance than
        available in ``init_conc``.

        Examples
        --------
        >>> rs = ReactionSystem.from_string('2 HNO2 -> H2O + NO + NO2 \\n 2 NO2 -> N2O4')
        >>> from collections import defaultdict
        >>> c0 = defaultdict(float, HNO2=20)
        >>> ref = {'HNO2': 20, 'H2O': 10, 'NO': 20, 'NO2': 20, 'N2O4': 10}
        >>> rs.as_per_substance_dict(rs.upper_conc_bounds(c0)) == ref
        True

        """
    import numpy as np
    if dtype is None:
        dtype = np.float64
    init_concs_arr = self.as_per_substance_array(init_concs, dtype=dtype)
    composition_conc = defaultdict(float)
    for conc, s_obj in zip(init_concs_arr, self.substances.values()):
        for comp_nr, coeff in s_obj.composition.items():
            if comp_nr in skip_keys:
                continue
            composition_conc[comp_nr] += coeff * conc
    bounds = []
    for s_obj in self.substances.values():
        choose_from = []
        for comp_nr, coeff in s_obj.composition.items():
            if comp_nr == 0:
                continue
            choose_from.append(composition_conc[comp_nr] / coeff)
        if len(choose_from) == 0:
            bounds.append(float('inf'))
        else:
            bounds.append(min_(choose_from))
    return bounds