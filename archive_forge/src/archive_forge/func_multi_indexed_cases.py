from collections import defaultdict, namedtuple, OrderedDict
from functools import wraps
from itertools import product
import os
import types
import warnings
from .. import __url__
from .deprecation import Deprecation
def multi_indexed_cases(od, *, dict_=OrderedDict, apply_keys=None, apply_values=None, apply_return=list, named_index=False):
    """Returns a list of length-2 tuples

    Each tuple consist of a multi-index (tuple of integers) and a dictionary.

    Parameters
    ----------
    od : OrderedDict
        Maps each key to a number of values. Instances of ``list``, ``tuple``,
        ``types.GeneratorType``, ``collections.abc.ItemsView`` are converted to ``OrderedDict``.
    dict_ : type, optional
        Used in the result (see ``Returns``).
    apply_keys : callable, optional
        Transformation of keys.
    apply_values : callable, optional
        Transformation of values.
    apply_return : callable, optional
        Applied on return value. ``None`` for generator.
    named_index : bool
        Tuple of indices will be a ``namedtuple`` (requires all keys to be ``str``).

    Examples
    --------
    >>> from chempy.util.pyutil import multi_indexed_cases
    >>> cases = multi_indexed_cases([('a', [1, 2, 3]), ('b', [False, True])])
    >>> len(cases)
    6
    >>> midxs, case_kws = zip(*cases)
    >>> midxs[0]
    (0, 0)
    >>> case_kws[0] == {'a': 1, 'b': False}
    True
    >>> d = {'a': 'foo bar'.split(), 'b': 'baz qux'.split()}
    >>> from chempy.util.pyutil import AttrDict
    >>> for nidx, case in multi_indexed_cases(d, dict_=AttrDict, named_index=True):
    ...     if case.a == 'bar' and case.b == 'baz':
    ...         print("{} {}".format(nidx.a, nidx.b))
    ...
    1 0


    Returns
    -------
    List of length-2 tuples, each consisting of one tuple of indices and one dictionary (of type ``dict_``).

    """
    if isinstance(od, OrderedDict):
        pass
    else:
        if hasattr(od, 'items'):
            od = od.items()
        if isinstance(od, (list, tuple, types.GeneratorType, ItemsView)):
            od = OrderedDict(od)
        else:
            raise NotImplementedError('Expected an OrderedDict')
    keys, values = (tuple(map(apply_keys or identity, od.keys())), tuple(od.values()))
    MultiIndex = namedtuple('MultiIndex', keys) if named_index else lambda *args: tuple(args)
    _generator = ((MultiIndex(*mi), dict_([(k, (apply_values or identity)(v[i])) for k, v, i in zip(keys, values, mi)])) for mi in product(*map(range, map(len, values))))
    return (apply_return or identity)(_generator)