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
def plot_dendrograms(self, row_cluster, col_cluster, metric, method, row_linkage, col_linkage, tree_kws):
    if row_cluster:
        self.dendrogram_row = dendrogram(self.data2d, metric=metric, method=method, label=False, axis=0, ax=self.ax_row_dendrogram, rotate=True, linkage=row_linkage, tree_kws=tree_kws)
    else:
        self.ax_row_dendrogram.set_xticks([])
        self.ax_row_dendrogram.set_yticks([])
    if col_cluster:
        self.dendrogram_col = dendrogram(self.data2d, metric=metric, method=method, label=False, axis=1, ax=self.ax_col_dendrogram, linkage=col_linkage, tree_kws=tree_kws)
    else:
        self.ax_col_dendrogram.set_xticks([])
        self.ax_col_dendrogram.set_yticks([])
    despine(ax=self.ax_row_dendrogram, bottom=True, left=True)
    despine(ax=self.ax_col_dendrogram, bottom=True, left=True)