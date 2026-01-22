import logging
def s_one_set(topics):
    """Perform s_one_set segmentation on a list of topics.
    Segmentation is defined as
    :math:`s_{set} = {(W', W^{*}) | W' = {w_i}; w_{i} \\in W; W^{*} = W}`

    Parameters
    ----------
    topics : list of `numpy.ndarray`
        List of topics obtained from an algorithm such as LDA.

    Returns
    -------
    list of list of (int, int).
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    --------
    .. sourcecode:: pycon

        >>> import numpy as np
        >>> from gensim.topic_coherence import segmentation
        >>>
        >>> topics = [np.array([9, 10, 7])]
        >>> segmentation.s_one_set(topics)
        [[(9, array([ 9, 10,  7])), (10, array([ 9, 10,  7])), (7, array([ 9, 10,  7]))]]

    """
    s_one_set_res = []
    for top_words in topics:
        s_one_set_t = []
        for w_prime in top_words:
            s_one_set_t.append((w_prime, top_words))
        s_one_set_res.append(s_one_set_t)
    return s_one_set_res