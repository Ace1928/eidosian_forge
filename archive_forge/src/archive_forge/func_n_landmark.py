from . import api
from . import base
from . import graphs
from . import matrix
from . import utils
from functools import partial
from scipy import sparse
import abc
import numpy as np
import pygsp
import tasklogger
@n_landmark.setter
def n_landmark(self, n_landmark):
    self._n_landmark = n_landmark
    utils.check_if_not(None, utils.check_positive, utils.check_int, n_landmark=n_landmark)
    self._update_n_landmark(n_landmark)