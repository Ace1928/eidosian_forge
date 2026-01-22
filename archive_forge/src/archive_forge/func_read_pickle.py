from . import base
from . import graphs
from scipy import sparse
import numpy as np
import pickle
import pygsp
import tasklogger
import warnings
def read_pickle(path):
    """Load pickled Graphtools object (or any object) from file.

    Parameters
    ----------
    path : str
        File path where the pickled object will be loaded.
    """
    with open(path, 'rb') as f:
        G = pickle.load(f)
    if not isinstance(G, base.BaseGraph):
        warnings.warn('Returning object that is not a graphtools.base.BaseGraph')
    elif isinstance(G, base.PyGSPGraph) and isinstance(G.logger, str):
        G.logger = pygsp.utils.build_logger(G.logger)
    return G