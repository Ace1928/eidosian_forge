import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def update_gamma(self):
    """Update variational dirichlet parameters.

        This operations is described in the original Blei LDA paper:
        gamma = alpha + sum(phi), over every topic for every word.

        Returns
        -------
        list of float
            The updated gamma parameters for each word in the document.

        """
    self.gamma = np.copy(self.lda.alpha)
    n = 0
    for word_id, count in self.doc:
        phi_row = self.phi[n]
        for k in range(self.lda.num_topics):
            self.gamma[k] += phi_row[k] * count
        n += 1
    return self.gamma