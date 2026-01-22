import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def generate_gensim_representation(self):
    """Create a gensim model from the stable topics.

        The returned representation is an Gensim LdaModel (:py:class:`gensim.models.LdaModel`) that has been
        instantiated with an A-priori belief on word probability, eta, that represents the topic-term distributions of
        any stable topics the were found by clustering over the ensemble of topic distributions.

        When no stable topics have been detected, None is returned.

        Returns
        -------
        :py:class:`gensim.models.LdaModel`
            A Gensim LDA Model classic_model_representation for which:
            ``classic_model_representation.get_topics() == self.get_topics()``

        """
    logger.info('generating classic gensim model representation based on results from the ensemble')
    sstats_sum = self.sstats_sum
    if sstats_sum == 0 and 'corpus' in self.gensim_kw_args and (not self.gensim_kw_args['corpus'] is None):
        for document in self.gensim_kw_args['corpus']:
            for token in document:
                sstats_sum += token[1]
        self.sstats_sum = sstats_sum
    stable_topics = self.get_topics()
    num_stable_topics = len(stable_topics)
    if num_stable_topics == 0:
        logger.error('the model did not detect any stable topic. You can try to adjust epsilon: recluster(eps=...)')
        self.classic_model_representation = None
        return
    params = self.gensim_kw_args.copy()
    params['eta'] = self.eta
    params['num_topics'] = num_stable_topics
    params['passes'] = 0
    classic_model_representation = self.get_topic_model_class()(**params)
    eta = classic_model_representation.eta
    if sstats_sum == 0:
        sstats_sum = classic_model_representation.state.sstats.sum()
        self.sstats_sum = sstats_sum
    eta_sum = 0
    if isinstance(eta, (int, float)):
        eta_sum = [eta * len(stable_topics[0])] * num_stable_topics
    else:
        if len(eta.shape) == 1:
            eta_sum = [[eta.sum()]] * num_stable_topics
        if len(eta.shape) > 1:
            eta_sum = np.array(eta.sum(axis=1)[:, None])
    normalization_factor = np.array([[sstats_sum / num_stable_topics]] * num_stable_topics) + eta_sum
    sstats = stable_topics * normalization_factor
    sstats -= eta
    classic_model_representation.state.sstats = sstats.astype(np.float32)
    classic_model_representation.sync_state()
    self.classic_model_representation = classic_model_representation
    return classic_model_representation