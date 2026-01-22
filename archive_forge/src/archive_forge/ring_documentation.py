import numpy as np
from scipy import sparse
from . import Graph  # prevent circular import in Python < 3.5
K-regular ring graph.

    Parameters
    ----------
    N : int
        Number of vertices.
    k : int
        Number of neighbors in each direction.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=10)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    