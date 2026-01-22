from typing import Any, Dict
import numpy as np
import cvxpy.interface as intf
def expcone_permutor(n_cones, exp_cone_order) -> np.ndarray:
    order = np.tile(np.array(exp_cone_order), n_cones)
    offsets = 3 * np.repeat(np.arange(n_cones), 3)
    perm = order + offsets
    return perm