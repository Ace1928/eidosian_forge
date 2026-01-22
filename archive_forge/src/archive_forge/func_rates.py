import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def rates(self, variables=None, backend=math, substance_keys=None, ratexs=None, cstr_fr_fc=None):
    """Per substance sums of reaction rates rates.

        Parameters
        ----------
        variables : dict
        backend : module, optional
        substance_keys : iterable of str, optional
        ratexs : iterable of RateExpr instances
        cstr_fr_fc : tuple (str, tuple of str)
            Continuously stirred tank reactor conditions. Pair of
            flow/volume ratio key (feed-rate/tank-volume) and dict mapping
            feed concentration keys to substance keys.

        Returns
        -------
        dict
            per substance_key time derivatives of concentrations.

        Examples
        --------
        >>> r = Reaction({'R': 2}, {'P': 1}, 42.0)
        >>> rsys = ReactionSystem([r])
        >>> rates = rsys.rates({'R': 3, 'P': 5})
        >>> abs(rates['P'] - 42*3**2) < 1e-14
        True

        """
    result = {}
    if ratexs is None:
        ratexs = [None] * self.nr
    for rxn, ratex in zip(self.rxns, ratexs):
        for k, v in rxn.rate(variables, backend, substance_keys, ratex=ratex).items():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
    if cstr_fr_fc:
        fr_key, fc = cstr_fr_fc
        for sk, fck in fc.items():
            result[sk] += variables[fr_key] * (variables[fck] - variables[sk])
    return result