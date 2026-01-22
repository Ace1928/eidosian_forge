from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def fromtextfile(fname, delimiter=None, commentchar='#', missingchar='', varnames=None, vartypes=None, *, delimitor=np._NoValue):
    """
    Creates a mrecarray from data stored in the file `filename`.

    Parameters
    ----------
    fname : {file name/handle}
        Handle of an opened file.
    delimiter : {None, string}, optional
        Alphanumeric character used to separate columns in the file.
        If None, any (group of) white spacestring(s) will be used.
    commentchar : {'#', string}, optional
        Alphanumeric character used to mark the start of a comment.
    missingchar : {'', string}, optional
        String indicating missing data, and used to create the masks.
    varnames : {None, sequence}, optional
        Sequence of the variable names. If None, a list will be created from
        the first non empty line of the file.
    vartypes : {None, sequence}, optional
        Sequence of the variables dtypes. If None, it will be estimated from
        the first non-commented line.


    Ultra simple: the varnames are in the header, one line"""
    if delimitor is not np._NoValue:
        if delimiter is not None:
            raise TypeError("fromtextfile() got multiple values for argument 'delimiter'")
        warnings.warn("The 'delimitor' keyword argument of numpy.ma.mrecords.fromtextfile() is deprecated since NumPy 1.22.0, use 'delimiter' instead.", DeprecationWarning, stacklevel=2)
        delimiter = delimitor
    ftext = openfile(fname)
    while True:
        line = ftext.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimiter)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames
    _variables = masked_array([line.strip().split(delimiter) for line in ftext if line[0] != commentchar and len(line) > 1])
    _, nfields = _variables.shape
    ftext.close()
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [np.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = 'Attempting to %i dtypes for %i fields!'
            msg += ' Reverting to default.'
            warnings.warn(msg % (len(vartypes), nfields), stacklevel=2)
            vartypes = _guessvartypes(_variables[0])
    mdescr = [(n, f) for n, f in zip(varnames, vartypes)]
    mfillv = [ma.default_fill_value(f) for f in vartypes]
    _mask = _variables.T == missingchar
    _datalist = [masked_array(a, mask=m, dtype=t, fill_value=f) for a, m, t, f in zip(_variables.T, _mask, vartypes, mfillv)]
    return fromarrays(_datalist, dtype=mdescr)