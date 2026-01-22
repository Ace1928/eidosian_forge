from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def plot_mts(Y, names=None, index=None):
    """
    Plot multiple time series
    """
    import matplotlib.pyplot as plt
    k = Y.shape[1]
    rows, cols = (k, 1)
    fig = plt.figure(figsize=(10, 10))
    for j in range(k):
        ts = Y[:, j]
        ax = fig.add_subplot(rows, cols, j + 1)
        if index is not None:
            ax.plot(index, ts)
        else:
            ax.plot(ts)
        if names is not None:
            ax.set_title(names[j])
    return fig