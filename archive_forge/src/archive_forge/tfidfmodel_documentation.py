import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
Get the tf-idf representation of an input vector and/or corpus.

        bow : {list of (int, int), iterable of iterable of (int, int)}
            Input document in the `sparse Gensim bag-of-words format
            <https://radimrehurek.com/gensim/intro.html#core-concepts>`_,
            or a streamed corpus of such documents.
        eps : float
            Threshold value, will remove all position that have tfidf-value less than `eps`.

        Returns
        -------
        vector : list of (int, float)
            TfIdf vector, if `bow` is a single document
        :class:`~gensim.interfaces.TransformedCorpus`
            TfIdf corpus, if `bow` is a corpus.

        