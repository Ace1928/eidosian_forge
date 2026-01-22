import os.path as op
import numpy as np
import networkx as nx
import pickle
from ... import logging
from ..base import (
from .base import have_cv

    Calculates and outputs the average network given a set of input NetworkX gpickle files

    See Also
    --------
    For documentation of Network-based statistic parameters:
    https://github.com/LTS5/connectomeviewer/blob/master/cviewer/libs/pyconto/groupstatistics/nbs/_nbs.py

    Example
    -------
    >>> import nipype.interfaces.cmtk as cmtk
    >>> nbs = cmtk.NetworkBasedStatistic()
    >>> nbs.inputs.in_group1 = ['subj1.pck', 'subj2.pck'] # doctest: +SKIP
    >>> nbs.inputs.in_group2 = ['pat1.pck', 'pat2.pck'] # doctest: +SKIP
    >>> nbs.run()                 # doctest: +SKIP

    