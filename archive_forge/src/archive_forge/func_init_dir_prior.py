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
def init_dir_prior(self, prior, name):
    """Initialize priors for the Dirichlet distribution.

        Parameters
        ----------
        prior : {float, numpy.ndarray of float, list of float, str}
            A-priori belief on document-topic distribution. If `name` == 'alpha', then the prior can be:
                * scalar for a symmetric prior over document-topic distribution,
                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,
                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).

            A-priori belief on topic-word distribution. If `name` == 'eta' then the prior can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'auto': Learns an asymmetric prior from the corpus.
        name : {'alpha', 'eta'}
            Whether the `prior` is parameterized by the alpha vector (1 parameter per topic)
            or by the eta (1 parameter per unique term in the vocabulary).

        Returns
        -------
        init_prior: numpy.ndarray
            Initialized Dirichlet prior:
            If 'alpha' was provided as `name` the shape is (self.num_topics, ).
            If 'eta' was provided as `name` the shape is (len(self.id2word), ).
        is_auto: bool
            Flag that shows if hyperparameter optimization should be used or not.
        """
    if prior is None:
        prior = 'symmetric'
    if name == 'alpha':
        prior_shape = self.num_topics
    elif name == 'eta':
        prior_shape = self.num_terms
    else:
        raise ValueError("'name' must be 'alpha' or 'eta'")
    is_auto = False
    if isinstance(prior, str):
        if prior == 'symmetric':
            logger.info('using symmetric %s at %s', name, 1.0 / self.num_topics)
            init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
        elif prior == 'asymmetric':
            if name == 'eta':
                raise ValueError("The 'asymmetric' option cannot be used for eta")
            init_prior = np.fromiter((1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
            init_prior /= init_prior.sum()
            logger.info('using asymmetric %s %s', name, list(init_prior))
        elif prior == 'auto':
            is_auto = True
            init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)), dtype=self.dtype, count=prior_shape)
            if name == 'alpha':
                logger.info('using autotuned %s, starting with %s', name, list(init_prior))
        else:
            raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
    elif isinstance(prior, list):
        init_prior = np.asarray(prior, dtype=self.dtype)
    elif isinstance(prior, np.ndarray):
        init_prior = prior.astype(self.dtype, copy=False)
    elif isinstance(prior, (np.number, numbers.Real)):
        init_prior = np.fromiter((prior for i in range(prior_shape)), dtype=self.dtype)
    else:
        raise ValueError('%s must be either a np array of scalars, list of scalars, or scalar' % name)
    return (init_prior, is_auto)