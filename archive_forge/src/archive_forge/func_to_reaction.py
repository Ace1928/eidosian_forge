from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def to_reaction(line, substance_keys, token, Cls, globals_=None, **kwargs):
    """Parses a string into a Reaction object and substances

    Reac1 + 2 Reac2 + (2 Reac1) -> Prod1 + Prod2; 10**3.7; ref='doi:12/ab'
    Reac1 = Prod1; 2.1;

    Parameters
    ----------
    line: str
        string representation to be parsed
    substance_keys: iterable of strings
        Allowed names, e.g. ('H2O', 'H+', 'OH-')
    token : str
        delimiter token between reactant and product side
    Cls : class
        e.g. subclass of Reaction
    globals_: dict (optional)
        Globals passed on to :func:`eval`, when ``None``:
        `chempy.units.default_units` is used with 'chempy'
        and 'default_units' extra entries.

    Notes
    -----
    This function calls :func:`eval`, hence there are severe security concerns
    with running this on untrusted data.

    """
    if globals_ is None:
        globals_ = get_parsing_context()
    parts = line.rstrip('\n').split(';')
    stoich = parts[0].strip()
    if len(parts) > 2:
        kwargs.update(eval('dict(' + ';'.join(parts[2:]) + '\n)', globals_ or {}))
    if len(parts) > 1:
        param = parts[1].strip()
    else:
        param = kwargs.pop('param', 'None')
    if isinstance(param, str):
        if param.startswith("'") and param.endswith("'") and ("'" not in param[1:-1]):
            from ..kinetics.rates import MassAction
            from ._expr import Symbol
            param = MassAction(Symbol(unique_keys=(param[1:-1],)))
        else:
            param = None if globals_ is False else eval(param, globals_)
    if token not in stoich:
        raise ValueError('Missing token: %s' % token)
    reac_prod = [[y.strip() for y in x.split(' + ')] for x in stoich.split(token)]
    act, inact = ([], [])
    for elements in reac_prod:
        act.append(_parse_multiplicity([x for x in elements if not x.startswith('(')], substance_keys))
        inact.append(_parse_multiplicity([x[1:-1] for x in elements if x.startswith('(') and x.endswith(')')], substance_keys))
    return Cls(act[0], act[1], param, inact_reac=inact[0], inact_prod=inact[1], **kwargs)