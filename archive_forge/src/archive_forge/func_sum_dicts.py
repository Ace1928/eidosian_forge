import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def sum_dicts(dicts):
    """Sums the dictionaries entrywise.

    Parameters
    ----------
    dicts : list
        A list of dictionaries with numeric entries.

    Returns
    -------
    dict
        A dict with the sum.
    """
    sum_dict = {}
    for val_dict in dicts:
        for id_, value in val_dict.items():
            if id_ in sum_dict:
                sum_dict[id_] = sum_dict[id_] + value
            else:
                sum_dict[id_] = value
    return sum_dict