import numpy as np
Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    start : List of tuples, each tuple of length equal to dimension of `ii`
        Coordinates of top left corner of window(s).
        Each tuple in the list contains the starting row, col, ... index
        i.e `[(row_win1, col_win1, ...), (row_win2, col_win2,...), ...]`.
    end : List of tuples, each tuple of length equal to dimension of `ii`
        Coordinates of bottom right corner of window(s).
        Each tuple in the list containing the end row, col, ... index i.e
        `[(row_win1, col_win1, ...), (row_win2, col_win2, ...), ...]`.

    Returns
    -------
    S : scalar or ndarray
        Integral (sum) over the given window(s).

    See Also
    --------
    integral_image : Create an integral image / summed area table.

    Examples
    --------
    >>> arr = np.ones((5, 6), dtype=float)
    >>> ii = integral_image(arr)
    >>> integrate(ii, (1, 0), (1, 2))  # sum from (1, 0) to (1, 2)
    array([3.])
    >>> integrate(ii, [(3, 3)], [(4, 5)])  # sum from (3, 3) to (4, 5)
    array([6.])
    >>> # sum from (1, 0) to (1, 2) and from (3, 3) to (4, 5)
    >>> integrate(ii, [(1, 0), (3, 3)], [(1, 2), (4, 5)])
    array([3., 6.])
    