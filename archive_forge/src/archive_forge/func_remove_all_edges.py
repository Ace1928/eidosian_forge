import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def remove_all_edges(ntwk):
    ntwktmp = ntwk.copy()
    edges = list(ntwktmp.edges())
    for edge in edges:
        ntwk.remove_edge(edge[0], edge[1])
    return ntwk