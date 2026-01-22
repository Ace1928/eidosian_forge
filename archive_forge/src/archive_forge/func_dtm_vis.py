import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def dtm_vis(self, time, corpus):
    """Get the information needed to visualize the corpus model at a given time slice, using the pyLDAvis format.

        Parameters
        ----------
        time : int
            The time slice we are interested in.
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            The corpus we want to visualize at the given time slice.

        Returns
        -------
        doc_topics : list of length `self.num_topics`
            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.
        topic_term : numpy.ndarray
            The representation of each topic as a multinomial over words in the vocabulary,
            expected shape (`num_topics`, vocabulary length).
        doc_lengths : list of int
            The number of words in each document. These could be fixed, or drawn from a Poisson distribution.
        term_frequency : numpy.ndarray
            The term frequency matrix (denoted as beta in the original Blei paper). This could also be the TF-IDF
            representation of the corpus, expected shape (number of documents, length of vocabulary).
        vocab : list of str
            The set of unique terms existing in the cropuse's vocabulary.

        """
    doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]

    def normalize(x):
        return x / x.sum()
    topic_term = [normalize(np.exp(chain.e_log_prob.T[time])) for k, chain in enumerate(self.topic_chains)]
    doc_lengths = []
    term_frequency = np.zeros(self.vocab_len)
    for doc_no, doc in enumerate(corpus):
        doc_lengths.append(len(doc))
        for term, freq in doc:
            term_frequency[term] += freq
    vocab = [self.id2word[i] for i in range(len(self.id2word))]
    return (doc_topic, np.array(topic_term), doc_lengths, term_frequency, vocab)