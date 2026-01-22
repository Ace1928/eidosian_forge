import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
@classmethod
def from_data(cls, m, title='Default title', key='0', mxtype=None, fmt=None):
    """Create a HBInfo instance from an existing sparse matrix.

        Parameters
        ----------
        m : sparse matrix
            the HBInfo instance will derive its parameters from m
        title : str
            Title to put in the HB header
        key : str
            Key
        mxtype : HBMatrixType
            type of the input matrix
        fmt : dict
            not implemented

        Returns
        -------
        hb_info : HBInfo instance
        """
    m = m.tocsc(copy=False)
    pointer = m.indptr
    indices = m.indices
    values = m.data
    nrows, ncols = m.shape
    nnon_zeros = m.nnz
    if fmt is None:
        pointer_fmt = IntFormat.from_number(np.max(pointer + 1))
        indices_fmt = IntFormat.from_number(np.max(indices + 1))
        if values.dtype.kind in np.typecodes['AllFloat']:
            values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
        elif values.dtype.kind in np.typecodes['AllInteger']:
            values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
        else:
            message = f'type {values.dtype.kind} not implemented yet'
            raise NotImplementedError(message)
    else:
        raise NotImplementedError('fmt argument not supported yet.')
    if mxtype is None:
        if not np.isrealobj(values):
            raise ValueError('Complex values not supported yet')
        if values.dtype.kind in np.typecodes['AllInteger']:
            tp = 'integer'
        elif values.dtype.kind in np.typecodes['AllFloat']:
            tp = 'real'
        else:
            raise NotImplementedError('type %s for values not implemented' % values.dtype)
        mxtype = HBMatrixType(tp, 'unsymmetric', 'assembled')
    else:
        raise ValueError('mxtype argument not handled yet.')

    def _nlines(fmt, size):
        nlines = size // fmt.repeat
        if nlines * fmt.repeat != size:
            nlines += 1
        return nlines
    pointer_nlines = _nlines(pointer_fmt, pointer.size)
    indices_nlines = _nlines(indices_fmt, indices.size)
    values_nlines = _nlines(values_fmt, values.size)
    total_nlines = pointer_nlines + indices_nlines + values_nlines
    return cls(title, key, total_nlines, pointer_nlines, indices_nlines, values_nlines, mxtype, nrows, ncols, nnon_zeros, pointer_fmt.fortran_format, indices_fmt.fortran_format, values_fmt.fortran_format)