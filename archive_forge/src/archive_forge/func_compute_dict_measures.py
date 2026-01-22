import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def compute_dict_measures(ntwk):
    """
    Returns a dictionary
    """
    iflogger.info('Computing measures which return a dictionary:')
    measures = {}
    iflogger.info('...Computing rich club coefficient...')
    measures['rich_club_coef'] = nx.rich_club_coefficient(ntwk)
    return measures