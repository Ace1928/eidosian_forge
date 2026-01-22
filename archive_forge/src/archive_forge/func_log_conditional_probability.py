import logging
import numpy as np
def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    """Calculate the log-conditional-probability measure which is used by coherence measures such as `U_mass`.
    This is defined as :math:`m_{lc}(S_i) = log \\frac{P(W', W^{*}) + \\epsilon}{P(W^{*})}`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, int)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,
        :func:`~gensim.topic_coherence.segmentation.s_one_one`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.
    with_std : bool, optional
        True to also include standard deviation across topic segment sets in addition to the mean coherence
        for each topic.
    with_support : bool, optional
        True to also include support across topic segments. The support is defined as the number of pairwise
        similarity comparisons were used to compute the overall topic coherence.

    Returns
    -------
    list of float
        Log conditional probabilities measurement for each topic.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis
        >>> from collections import namedtuple
        >>>
        >>> # Create dictionary
        >>> id2token = {1: 'test', 2: 'doc'}
        >>> token2id = {v: k for k, v in id2token.items()}
        >>> dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
        >>>
        >>> # Initialize segmented topics and accumulator
        >>> segmentation = [[(1, 2)]]
        >>>
        >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        >>> accumulator._num_docs = 5
        >>>
        >>> # result should be ~ ln(1 / 2) = -0.693147181
        >>> result = direct_confirmation_measure.log_conditional_probability(segmentation, accumulator)[0]

    """
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            try:
                w_star_count = accumulator[w_star]
                co_occur_count = accumulator[w_prime, w_star]
                m_lc_i = np.log((co_occur_count / num_docs + EPSILON) / (w_star_count / num_docs))
            except KeyError:
                m_lc_i = 0.0
            except ZeroDivisionError:
                m_lc_i = 0.0
            segment_sims.append(m_lc_i)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences