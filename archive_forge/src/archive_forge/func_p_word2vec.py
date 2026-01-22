import itertools
import logging
from gensim.topic_coherence.text_analysis import (
def p_word2vec(texts, segmented_topics, dictionary, window_size=None, processes=1, model=None):
    """Train word2vec model on `texts` if `model` is not None.

    Parameters
    ----------
    texts : iterable of iterable of str
        Input text
    segmented_topics : iterable of iterable of str
        Output from the segmentation of topics. Could be simply topics too.
    dictionary : :class:`~gensim.corpora.dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int, optional
        Size of the sliding window.
    processes : int, optional
        Number of processes to use.
    model : :class:`~gensim.models.word2vec.Word2Vec` or :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
        If None, a new Word2Vec model is trained on the given text corpus. Otherwise,
        it should be a pre-trained Word2Vec context vectors.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator`
        Text accumulator with trained context vectors.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.topic_coherence import probability_estimation
        >>> from gensim.corpora.hashdictionary import HashDictionary
        >>> from gensim.models import word2vec
        >>>
        >>> texts = [
        ...     ['human', 'interface', 'computer'],
        ...     ['eps', 'user', 'interface', 'system'],
        ...     ['system', 'human', 'system', 'eps'],
        ...     ['user', 'response', 'time'],
        ...     ['trees'],
        ...     ['graph', 'trees']
        ... ]
        >>> dictionary = HashDictionary(texts)
        >>> w2id = dictionary.token2id

        >>>
        >>> # create segmented_topics
        >>> segmented_topics = [
        ...     [
        ...         (w2id['system'], w2id['graph']),
        ...         (w2id['computer'], w2id['graph']),
        ...         (w2id['computer'], w2id['system'])
        ...     ],
        ...     [
        ...         (w2id['computer'], w2id['graph']),
        ...         (w2id['user'], w2id['graph']),
        ...         (w2id['user'], w2id['computer'])]
        ... ]
        >>> # create corpus
        >>> corpus = [dictionary.doc2bow(text) for text in texts]
        >>> sentences = [
        ...     ['human', 'interface', 'computer'],
        ...     ['survey', 'user', 'computer', 'system', 'response', 'time']
        ... ]
        >>> model = word2vec.Word2Vec(sentences, vector_size=100, min_count=1)
        >>> accumulator = probability_estimation.p_word2vec(texts, segmented_topics, dictionary, 2, 1, model)

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)