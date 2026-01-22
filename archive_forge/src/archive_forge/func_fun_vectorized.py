import numpy as np
def fun_vectorized(t, y):
    f = np.empty_like(y)
    for i, yi in enumerate(y.T):
        f[:, i] = self._fun(t, yi)
    return f