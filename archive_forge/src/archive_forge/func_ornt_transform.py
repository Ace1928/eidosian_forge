import numpy as np
import numpy.linalg as npl
from .deprecated import deprecate_with_version
def ornt_transform(start_ornt, end_ornt):
    """Return the orientation that transforms from `start_ornt` to `end_ornt`.

    Parameters
    ----------
    start_ornt : (n,2) orientation array
        Initial orientation.

    end_ornt : (n,2) orientation array
        Final orientation.

    Returns
    -------
    orientations : (p, 2) ndarray
       The orientation that will transform the `start_ornt` to the `end_ornt`.
    """
    start_ornt = np.asarray(start_ornt)
    end_ornt = np.asarray(end_ornt)
    if start_ornt.shape != end_ornt.shape:
        raise ValueError('The orientations must have the same shape')
    if start_ornt.shape[1] != 2:
        raise ValueError(f'Invalid shape for an orientation: {start_ornt.shape}')
    result = np.empty_like(start_ornt)
    for end_in_idx, (end_out_idx, end_flip) in enumerate(end_ornt):
        for start_in_idx, (start_out_idx, start_flip) in enumerate(start_ornt):
            if end_out_idx == start_out_idx:
                if start_flip == end_flip:
                    flip = 1
                else:
                    flip = -1
                result[start_in_idx, :] = [end_in_idx, flip]
                break
        else:
            raise ValueError('Unable to find out axis %d in start_ornt' % end_out_idx)
    return result