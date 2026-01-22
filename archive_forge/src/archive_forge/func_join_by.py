import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_join_by_dispatcher)
def join_by(key, r1, r2, jointype='inner', r1postfix='1', r2postfix='2', defaults=None, usemask=True, asrecarray=False):
    """
    Join arrays `r1` and `r2` on key `key`.

    The key should be either a string or a sequence of string corresponding
    to the fields used to join the array.  An exception is raised if the
    `key` field cannot be found in the two input arrays.  Neither `r1` nor
    `r2` should have any duplicates along `key`: the presence of duplicates
    will make the output quite unreliable. Note that duplicates are not
    looked for by the algorithm.

    Parameters
    ----------
    key : {string, sequence}
        A string or a sequence of strings corresponding to the fields used
        for comparison.
    r1, r2 : arrays
        Structured arrays.
    jointype : {'inner', 'outer', 'leftouter'}, optional
        If 'inner', returns the elements common to both r1 and r2.
        If 'outer', returns the common elements as well as the elements of
        r1 not in r2 and the elements of not in r2.
        If 'leftouter', returns the common elements and the elements of r1
        not in r2.
    r1postfix : string, optional
        String appended to the names of the fields of r1 that are present
        in r2 but absent of the key.
    r2postfix : string, optional
        String appended to the names of the fields of r2 that are present
        in r1 but absent of the key.
    defaults : {dictionary}, optional
        Dictionary mapping field names to the corresponding default values.
    usemask : {True, False}, optional
        Whether to return a MaskedArray (or MaskedRecords is
        `asrecarray==True`) or a ndarray.
    asrecarray : {False, True}, optional
        Whether to return a recarray (or MaskedRecords if `usemask==True`)
        or just a flexible-type ndarray.

    Notes
    -----
    * The output is sorted along the key.
    * A temporary array is formed by dropping the fields not in the key for
      the two arrays and concatenating the result. This array is then
      sorted, and the common entries selected. The output is constructed by
      filling the fields with the selected entries. Matching is not
      preserved if there are some duplicates...

    """
    if jointype not in ('inner', 'outer', 'leftouter'):
        raise ValueError("The 'jointype' argument should be in 'inner', 'outer' or 'leftouter' (got '%s' instead)" % jointype)
    if isinstance(key, str):
        key = (key,)
    if len(set(key)) != len(key):
        dup = next((x for n, x in enumerate(key) if x in key[n + 1:]))
        raise ValueError('duplicate join key %r' % dup)
    for name in key:
        if name not in r1.dtype.names:
            raise ValueError('r1 does not have key field %r' % name)
        if name not in r2.dtype.names:
            raise ValueError('r2 does not have key field %r' % name)
    r1 = r1.ravel()
    r2 = r2.ravel()
    nb1 = len(r1)
    r1names, r2names = (r1.dtype.names, r2.dtype.names)
    collisions = (set(r1names) & set(r2names)) - set(key)
    if collisions and (not (r1postfix or r2postfix)):
        msg = 'r1 and r2 contain common names, r1postfix and r2postfix '
        msg += "can't both be empty"
        raise ValueError(msg)
    key1 = [n for n in r1names if n in key]
    r1k = _keep_fields(r1, key1)
    r2k = _keep_fields(r2, key1)
    aux = ma.concatenate((r1k, r2k))
    idx_sort = aux.argsort(order=key)
    aux = aux[idx_sort]
    flag_in = ma.concatenate(([False], aux[1:] == aux[:-1]))
    flag_in[:-1] = flag_in[1:] + flag_in[:-1]
    idx_in = idx_sort[flag_in]
    idx_1 = idx_in[idx_in < nb1]
    idx_2 = idx_in[idx_in >= nb1] - nb1
    r1cmn, r2cmn = (len(idx_1), len(idx_2))
    if jointype == 'inner':
        r1spc, r2spc = (0, 0)
    elif jointype == 'outer':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[idx_out < nb1]))
        idx_2 = np.concatenate((idx_2, idx_out[idx_out >= nb1] - nb1))
        r1spc, r2spc = (len(idx_1) - r1cmn, len(idx_2) - r2cmn)
    elif jointype == 'leftouter':
        idx_out = idx_sort[~flag_in]
        idx_1 = np.concatenate((idx_1, idx_out[idx_out < nb1]))
        r1spc, r2spc = (len(idx_1) - r1cmn, 0)
    s1, s2 = (r1[idx_1], r2[idx_2])
    ndtype = _get_fieldspec(r1k.dtype)
    for fname, fdtype in _get_fieldspec(r1.dtype):
        if fname not in key:
            ndtype.append((fname, fdtype))
    for fname, fdtype in _get_fieldspec(r2.dtype):
        names = list((name for name, dtype in ndtype))
        try:
            nameidx = names.index(fname)
        except ValueError:
            ndtype.append((fname, fdtype))
        else:
            _, cdtype = ndtype[nameidx]
            if fname in key:
                ndtype[nameidx] = (fname, max(fdtype, cdtype))
            else:
                ndtype[nameidx:nameidx + 1] = [(fname + r1postfix, cdtype), (fname + r2postfix, fdtype)]
    ndtype = np.dtype(ndtype)
    cmn = max(r1cmn, r2cmn)
    output = ma.masked_all((cmn + r1spc + r2spc,), dtype=ndtype)
    names = output.dtype.names
    for f in r1names:
        selected = s1[f]
        if f not in names or (f in r2names and (not r2postfix) and (f not in key)):
            f += r1postfix
        current = output[f]
        current[:r1cmn] = selected[:r1cmn]
        if jointype in ('outer', 'leftouter'):
            current[cmn:cmn + r1spc] = selected[r1cmn:]
    for f in r2names:
        selected = s2[f]
        if f not in names or (f in r1names and (not r1postfix) and (f not in key)):
            f += r2postfix
        current = output[f]
        current[:r2cmn] = selected[:r2cmn]
        if jointype == 'outer' and r2spc:
            current[-r2spc:] = selected[r2cmn:]
    output.sort(order=key)
    kwargs = dict(usemask=usemask, asrecarray=asrecarray)
    return _fix_output(_fix_defaults(output, defaults), **kwargs)