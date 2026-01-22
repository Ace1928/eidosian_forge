from ..libmp.backend import xrange
from .calculus import defun
def dpoly(n, _cache={}):
    """
    nth differentiation polynomial for exp (Faa di Bruno's formula).

    TODO: most exponents are zero, so maybe a sparse representation
    would be better.
    """
    if n in _cache:
        return _cache[n]
    if not _cache:
        _cache[0] = {(0,): 1}
    R = dpoly(n - 1)
    R = dict(((c + (0,), v) for c, v in iteritems(R)))
    Ra = {}
    for powers, count in iteritems(R):
        powers1 = (powers[0] + 1,) + powers[1:]
        if powers1 in Ra:
            Ra[powers1] += count
        else:
            Ra[powers1] = count
    for powers, count in iteritems(R):
        if not sum(powers):
            continue
        for k, p in enumerate(powers):
            if p:
                powers2 = powers[:k] + (p - 1, powers[k + 1] + 1) + powers[k + 2:]
                if powers2 in Ra:
                    Ra[powers2] += p * count
                else:
                    Ra[powers2] = p * count
    _cache[n] = Ra
    return _cache[n]