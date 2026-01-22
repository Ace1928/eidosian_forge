from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def compare_models(self, models):
    """Compare topic models by coherence value.

        Parameters
        ----------
        models : :class:`~gensim.models.basemodel.BaseTopicModel`
            Sequence of topic models.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        """
    model_topics = [self._get_topics_from_model(model, self.topn) for model in models]
    return self.compare_model_topics(model_topics)