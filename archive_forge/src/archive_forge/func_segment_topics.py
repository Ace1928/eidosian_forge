from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def segment_topics(self):
    """Segment topic, alias for `self.measure.seg(self.topics)`.

        Return
        ------
        list of list of pair
            Segmented topics.

        """
    return self.measure.seg(self.topics)