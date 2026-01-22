from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def estimate_probabilities(self, segmented_topics=None):
    """Accumulate word occurrences and co-occurrences from texts or corpus using the optimal method for the chosen
        coherence metric.

        Notes
        -----
        This operation may take quite some time for the sliding window based coherence methods.

        Parameters
        ----------
        segmented_topics : list of list of pair, optional
            Segmented topics, typically produced by :meth:`~gensim.models.coherencemodel.CoherenceModel.segment_topics`.

        Return
        ------
        :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`
            Corpus accumulator.

        """
    if segmented_topics is None:
        segmented_topics = self.segment_topics()
    if self.coherence in BOOLEAN_DOCUMENT_BASED:
        self._accumulator = self.measure.prob(self.corpus, segmented_topics)
    else:
        kwargs = dict(texts=self.texts, segmented_topics=segmented_topics, dictionary=self.dictionary, window_size=self.window_size, processes=self.processes)
        if self.coherence == 'c_w2v':
            kwargs['model'] = self.keyed_vectors
        self._accumulator = self.measure.prob(**kwargs)
    return self._accumulator