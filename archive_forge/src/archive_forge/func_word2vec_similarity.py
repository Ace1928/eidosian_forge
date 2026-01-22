import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def word2vec_similarity(segmented_topics, accumulator, with_std=False, with_support=False):
    """For each topic segmentation, compute average cosine similarity using a
    :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, `numpy.ndarray`)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_set`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator` or
                  :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator.
    with_std : bool, optional
        True to also include standard deviation across topic segment sets
        in addition to the mean coherence for each topic.
    with_support : bool, optional
        True to also include support across topic segments. The support is defined as
        the number of pairwise similarity comparisons were used to compute the overall topic coherence.

    Returns
    -------
    list of (float[, float[, int]])
        Сosine word2vec similarities per topic (with std/support if `with_std`, `with_support`).

    Examples
    --------
    .. sourcecode:: pycon

        >>> import numpy as np
        >>> from gensim.corpora.dictionary import Dictionary
        >>> from gensim.topic_coherence import indirect_confirmation_measure
        >>> from gensim.topic_coherence import text_analysis
        >>>
        >>> # create segmentation
        >>> segmentation = [[(1, np.array([1, 2])), (2, np.array([1, 2]))]]
        >>>
        >>> # create accumulator
        >>> dictionary = Dictionary()
        >>> dictionary.id2token = {1: 'fake', 2: 'tokens'}
        >>> accumulator = text_analysis.WordVectorsAccumulator({1, 2}, dictionary)
        >>> _ = accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
        >>>
        >>> # should be (0.726752426218 0.00695475919227)
        >>> mean, std = indirect_confirmation_measure.word2vec_similarity(segmentation, accumulator, with_std=True)[0]

    """
    topic_coherences = []
    total_oov = 0
    for topic_index, topic_segments in enumerate(segmented_topics):
        segment_sims = []
        num_oov = 0
        for w_prime, w_star in topic_segments:
            if not hasattr(w_prime, '__iter__'):
                w_prime = [w_prime]
            if not hasattr(w_star, '__iter__'):
                w_star = [w_star]
            try:
                segment_sims.append(accumulator.ids_similarity(w_prime, w_star))
            except ZeroDivisionError:
                num_oov += 1
        if num_oov > 0:
            total_oov += 1
            logger.warning('%d terms for topic %d are not in word2vec model vocabulary', num_oov, topic_index)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    if total_oov > 0:
        logger.warning('%d terms for are not in word2vec model vocabulary', total_oov)
    return topic_coherences