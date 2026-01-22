from __future__ import division
import sys
import importlib
import logging
import functools
import pkgutil
import io
import numpy as np
from scipy import sparse
import scipy.io
def graph_array_handler(func):

    def inner(G, *args, **kwargs):
        if type(G) is list:
            output = []
            for g in G:
                output.append(func(g, *args, **kwargs))
            return output
        else:
            return func(G, *args, **kwargs)
    return inner