from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def define_bin_params(self, x1, x2=None, weights=None, cache=True):
    """Given data, return numpy.histogram parameters to define bins."""
    if x2 is None:
        bin_edges = self._define_bin_edges(x1, weights, self.bins, self.binwidth, self.binrange, self.discrete)
        if isinstance(self.bins, (str, Number)):
            n_bins = len(bin_edges) - 1
            bin_range = (bin_edges.min(), bin_edges.max())
            bin_kws = dict(bins=n_bins, range=bin_range)
        else:
            bin_kws = dict(bins=bin_edges)
    else:
        bin_edges = []
        for i, x in enumerate([x1, x2]):
            bins = self.bins
            if not bins or isinstance(bins, (str, Number)):
                pass
            elif isinstance(bins[i], str):
                bins = bins[i]
            elif len(bins) == 2:
                bins = bins[i]
            binwidth = self.binwidth
            if binwidth is None:
                pass
            elif not isinstance(binwidth, Number):
                binwidth = binwidth[i]
            binrange = self.binrange
            if binrange is None:
                pass
            elif not isinstance(binrange[0], Number):
                binrange = binrange[i]
            discrete = self.discrete
            if not isinstance(discrete, bool):
                discrete = discrete[i]
            bin_edges.append(self._define_bin_edges(x, weights, bins, binwidth, binrange, discrete))
        bin_kws = dict(bins=tuple(bin_edges))
    if cache:
        self.bin_kws = bin_kws
    return bin_kws