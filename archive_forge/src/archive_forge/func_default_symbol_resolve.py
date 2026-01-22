from rpy2 import rinterface
from warnings import warn
from collections import defaultdict
def default_symbol_resolve(symbol_mapping):
    """Resolve any conflict in a symbol mapping.

    The argument `symbol_mapping` maps candidate new symbol names
    (e.g., the names of Python attributes in the namespace returned by
    :func:`importr`)
    to a sequence of original symbol names (e.g., the names of objects in
    an R package). The purpose of this function is to resolved conflicts,
    that is situations where there is more than one original symbol name
    associated with a new symbol name.

    :param symbol_mapping: a :class:`dict` or dict-like object.
    :return: A 2-tuple with conflicts (a :class:`dict` mapping the new
    symbol to a sequence of symbols already matching) and resolutions (a
    :class:`dict` mapping new symbol).
    """
    conflicts = dict()
    resolutions = dict()
    for py_symbol, r_symbols in symbol_mapping.items():
        n_r_symbols = len(r_symbols)
        if n_r_symbols == 1:
            continue
        elif n_r_symbols == 2:
            try:
                idx = r_symbols.index(py_symbol)
                for i, r_name in enumerate(r_symbols):
                    if i == idx:
                        resolutions[py_symbol] = [r_name]
                    else:
                        new_py_symbol = py_symbol + '_'
                        resolutions[new_py_symbol] = [r_name]
            except ValueError:
                conflicts[py_symbol] = r_symbols
        else:
            conflicts[py_symbol] = r_symbols
    return (conflicts, resolutions)