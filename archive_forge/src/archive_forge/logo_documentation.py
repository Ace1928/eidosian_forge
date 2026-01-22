import numpy as np
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
GSP logo.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Logo()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(ax=axes[1])

    