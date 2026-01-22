from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
def format_topic(self, topic_id, topic_terms):
    """Format the display for a single topic in two different ways.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topic_terms : list of (str, numpy.float)
            Contains the most probable words from a single topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for topic terms depends on the value of `self.style` attribute.

        """
    if self.STYLE_GENSIM == self.style:
        fmt = ' + '.join(('%.3f*%s' % (weight, word) for word, weight in topic_terms))
    else:
        fmt = '\n'.join(('    %20s    %.8f' % (word, weight) for word, weight in topic_terms))
    fmt = (topic_id, fmt)
    return fmt