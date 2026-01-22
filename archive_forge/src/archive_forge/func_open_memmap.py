import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def open_memmap(filename, mode='r+', dtype=None, shape=None, fortran_order=False, version=None, *, max_header_size=_MAX_HEADER_SIZE):
    """
    Open a .npy file as a memory-mapped array.

    This may be used to read an existing file or create a new one.

    Parameters
    ----------
    filename : str or path-like
        The name of the file on disk.  This may *not* be a file-like
        object.
    mode : str, optional
        The mode in which to open the file; the default is 'r+'.  In
        addition to the standard file modes, 'c' is also accepted to mean
        "copy on write."  See `memmap` for the available mode strings.
    dtype : data-type, optional
        The data type of the array if we are creating a new file in "write"
        mode, if not, `dtype` is ignored.  The default value is None, which
        results in a data-type of `float64`.
    shape : tuple of int
        The shape of the array if we are creating a new file in "write"
        mode, in which case this parameter is required.  Otherwise, this
        parameter is ignored and is thus optional.
    fortran_order : bool, optional
        Whether the array should be Fortran-contiguous (True) or
        C-contiguous (False, the default) if we are creating a new file in
        "write" mode.
    version : tuple of int (major, minor) or None
        If the mode is a "write" mode, then this is the version of the file
        format used to create the file.  None means use the oldest
        supported version that is able to store the data.  Default: None
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    marray : memmap
        The memory-mapped array.

    Raises
    ------
    ValueError
        If the data or the mode is invalid.
    OSError
        If the file is not found or cannot be opened correctly.

    See Also
    --------
    numpy.memmap

    """
    if isfileobj(filename):
        raise ValueError('Filename must be a string or a path-like object.  Memmap cannot use existing file handles.')
    if 'w' in mode:
        _check_version(version)
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = dict(descr=dtype_to_descr(dtype), fortran_order=fortran_order, shape=shape)
        with open(os_fspath(filename), mode + 'b') as fp:
            _write_array_header(fp, d, version)
            offset = fp.tell()
    else:
        with open(os_fspath(filename), 'rb') as fp:
            version = read_magic(fp)
            _check_version(version)
            shape, fortran_order, dtype = _read_array_header(fp, version, max_header_size=max_header_size)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = fp.tell()
    if fortran_order:
        order = 'F'
    else:
        order = 'C'
    if mode == 'w+':
        mode = 'r+'
    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order, mode=mode, offset=offset)
    return marray