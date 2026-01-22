import numpy as np
import numpy.linalg as npl
from .deprecated import deprecate_with_version
def axcodes2ornt(axcodes, labels=None):
    """Convert axis codes `axcodes` to an orientation

    Parameters
    ----------
    axcodes : (N,) tuple
        axis codes - see ornt2axcodes docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first element in `axcodes` is ``front``, and the second
        (2,) sequence in `labels` is ('back', 'front') then the first
        row of `ornt` will be ``[1, 1]``. If None, equivalent to
        ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

    Returns
    -------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring

    Examples
    --------
    >>> axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    array([[ 1.,  1.],
           [ 0., -1.],
           [ 2.,  1.]])
    """
    labels = list(zip('LPI', 'RAS')) if labels is None else labels
    allowed_labels = sum([list(L) for L in labels], []) + [None]
    if len(allowed_labels) != len(set(allowed_labels)):
        raise ValueError(f'Duplicate labels in {allowed_labels}')
    if not set(axcodes).issubset(allowed_labels):
        raise ValueError(f'Not all axis codes {list(axcodes)} in label set {allowed_labels}')
    n_axes = len(axcodes)
    ornt = np.ones((n_axes, 2), dtype=np.int8) * np.nan
    for code_idx, code in enumerate(axcodes):
        for label_idx, codes in enumerate(labels):
            if code is None:
                continue
            if code in codes:
                if code == codes[0]:
                    ornt[code_idx, :] = [label_idx, -1]
                else:
                    ornt[code_idx, :] = [label_idx, 1]
                break
    return ornt