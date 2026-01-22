import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def update_phi(self, doc_number, time):
    """Update variational multinomial parameters, based on a document and a time-slice.

        This is done based on the original Blei-LDA paper, where:
        log_phi := beta * exp(Ψ(gamma)), over every topic for every word.

        TODO: incorporate lee-sueng trick used in
        **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        Parameters
        ----------
        doc_number : int
            Document number. Unused.
        time : int
            Time slice. Unused.

        Returns
        -------
        (list of float, list of float)
            Multinomial parameters, and their logarithm, for each word in the document.

        """
    num_topics = self.lda.num_topics
    dig = np.zeros(num_topics)
    for k in range(num_topics):
        dig[k] = digamma(self.gamma[k])
    n = 0
    for word_id, count in self.doc:
        for k in range(num_topics):
            self.log_phi[n][k] = dig[k] + self.lda.topics[word_id][k]
        log_phi_row = self.log_phi[n]
        phi_row = self.phi[n]
        v = log_phi_row[0]
        for i in range(1, len(log_phi_row)):
            v = np.logaddexp(v, log_phi_row[i])
        log_phi_row = log_phi_row - v
        phi_row = np.exp(log_phi_row)
        self.log_phi[n] = log_phi_row
        self.phi[n] = phi_row
        n += 1
    return (self.phi, self.log_phi)