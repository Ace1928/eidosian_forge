import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
@property
def reordered_ind(self):
    """Indices of the matrix, reordered by the dendrogram"""
    return self.dendrogram['leaves']