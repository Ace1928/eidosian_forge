import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
Random sensor graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 64)
    Nc : int
        Minimum number of connections (default = 2)
    regular : bool
        Flag to fix the number of connections to nc (default = False)
    n_try : int
        Number of attempt to create the graph (default = 50)
    distribute : bool
        To distribute the points more evenly (default = False)
    connected : bool
        To force the graph to be connected (default = True)
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sensor(N=64, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    