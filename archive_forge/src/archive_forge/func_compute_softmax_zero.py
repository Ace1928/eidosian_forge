import numpy as np
def compute_softmax_zero(values: np.ndarray) -> np.ndarray:
    """The function modifies the input inplace."""
    v_max = values.max()
    exp_neg_v_max = np.exp(-v_max)
    s = 0
    for i in range(len(values)):
        v = values[i]
        if v > 1e-07 or v < -1e-07:
            values[i] = np.exp(v - v_max)
        else:
            values[i] *= exp_neg_v_max
        s += values[i]
    if s == 0:
        values[:] = 0.5
    else:
        values[:] /= s
    return values