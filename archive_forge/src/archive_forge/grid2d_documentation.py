import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
2-dimensional grid graph.

    Parameters
    ----------
    N1 : int
        Number of vertices along the first dimension.
    N2 : int
        Number of vertices along the second dimension (default N1).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Grid2d(N1=5, N2=4)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    