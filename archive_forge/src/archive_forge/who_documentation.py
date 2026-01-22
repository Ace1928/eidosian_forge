import sys
import cupy
Print the CuPy arrays in the given dictionary.

    Prints out the name, shape, bytes and type of all of the ndarrays
    present in `vardict`.

    If there is no dictionary passed in or `vardict` is None then returns
    CuPy arrays in the globals() dictionary (all CuPy arrays in the
    namespace).

    Args:
        vardict : (None or dict)  A dictionary possibly containing ndarrays.
                  Default is globals() if `None` specified


    .. admonition:: Example

        >>> a = cupy.arange(10)
        >>> b = cupy.ones(20)
        >>> cupy.who()
        Name            Shape            Bytes            Type
        ===========================================================
        <BLANKLINE>
        a               10               80               int64
        b               20               160              float64
        <BLANKLINE>
        Upper bound on total bytes  =       240
        >>> d = {'x': cupy.arange(2.0),
        ... 'y': cupy.arange(3.0), 'txt': 'Some str',
        ... 'idx':5}
        >>> cupy.who(d)
        Name            Shape            Bytes            Type
        ===========================================================
        <BLANKLINE>
        x               2                16               float64
        y               3                24               float64
        <BLANKLINE>
        Upper bound on total bytes  =       40

    