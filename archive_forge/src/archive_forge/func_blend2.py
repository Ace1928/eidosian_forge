import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def blend2(self, rhot, other, targetsize=None):
    """Merge the current state with another one using a weighted sum for the sufficient statistics.

        In contrast to :meth:`~gensim.models.ldamodel.LdaState.blend`, the sufficient statistics are not scaled
        prior to aggregation.

        Parameters
        ----------
        rhot : float
            Unused.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
    assert other is not None
    if targetsize is None:
        targetsize = self.numdocs
    self.sstats += other.sstats
    self.numdocs = targetsize