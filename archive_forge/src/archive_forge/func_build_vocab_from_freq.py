import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
    """Build vocabulary from a dictionary of word frequencies.

        Build model vocabulary from a passed dictionary that contains a (word -> word count) mapping.
        Words must be of type unicode strings.

        Parameters
        ----------
        word_freq : dict of (str, int)
            Word <-> count mapping.
        keep_raw_vocab : bool, optional
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        corpus_count : int, optional
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        update : bool, optional
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        """
    logger.info('processing provided word frequencies')
    raw_vocab = word_freq
    logger.info('collected %i different raw words, with total frequency of %i', len(raw_vocab), sum(raw_vocab.values()))
    self.corpus_count = corpus_count or 0
    self.raw_vocab = raw_vocab
    report_values = self.prepare_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
    report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
    self.prepare_weights(update=update)