import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
def show_topics(self, num_topics=-1, num_words=10, log=False, formatted=True):
    """Get the most significant topics.

        Parameters
        ----------
        num_topics : int, optional
            The number of topics to be selected, if -1 - all topics will be in result (ordered by significance).
        num_words : int, optional
            The number of words to be included per topics (ordered by significance).
        log : bool, optional
            If True - log topics with logger.
        formatted : bool, optional
            If True - each topic represented as string, otherwise - in BoW format.

        Returns
        -------
        list of (int, str)
            If `formatted=True`, return sequence with (topic_id, string representation of topics) **OR**
        list of (int, list of (str, float))
            Otherwise, return sequence with (topic_id, [(word, value), ... ]).

        """
    shown = []
    if num_topics < 0:
        num_topics = self.num_topics
    for i in range(min(num_topics, self.num_topics)):
        if i < len(self.projection.s):
            if formatted:
                topic = self.print_topic(i, topn=num_words)
            else:
                topic = self.show_topic(i, topn=num_words)
            shown.append((i, topic))
            if log:
                logger.info('topic #%i(%.3f): %s', i, self.projection.s[i], topic)
    return shown