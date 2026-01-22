from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def ionic_strength(molalities, charges=None, units=None, substances=None, substance_factory=Substance.from_formula, warn=True):
    """Calculates the ionic strength

    Parameters
    ----------
    molalities: array_like or dict
        Optionally with unit (amount / mass).
        when dict: mapping substance key to molality.
    charges: array_like
        Charge of respective ion, taken for substances when None.
    units: object (optional, default: None)
        Attributes accessed: molal.
    substances: dict, optional
        Mapping of substance keys to Substance instances (used when molalities
        is a dict).
    substance_factory: callback
        Used if `substances` is a string.
    warn: bool
        Issue a warning if molalities violates net charge neutrality.

    Examples
    --------
    >>> ionic_strength([1e-3, 3e-3], [3, -1]) == .5 * (9 + 3) * 1e-3
    True
    >>> ionic_strength({'Mg+2': 6, 'PO4-3': 4})
    30.0

    """
    tot = None
    if charges is None:
        if substances is None:
            substances = ' '.join(molalities.keys())
        if isinstance(substances, str):
            substances = OrderedDict([(k, substance_factory(k)) for k in substances.split()])
        charges, molalities = zip(*[(substances[k].charge, v) for k, v in molalities.items()])
    if len(molalities) != len(charges):
        raise ValueError('molalities and charges of different lengths')
    for b, z in zip(molalities, charges):
        if tot is None:
            tot = b * z ** 2
        else:
            tot += b * z ** 2
    if warn:
        net = None
        for b, z in zip(molalities, charges):
            if net is None:
                net = b * z
            else:
                net += b * z
        if not allclose(net, tot * 0, atol=tot * 1e-14):
            warnings.warn('Molalities not charge neutral: %s' % str(net))
    return tot / 2