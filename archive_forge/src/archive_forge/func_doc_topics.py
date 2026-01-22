import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def doc_topics(self, doc_number):
    """Get the topic mixture for a document.

        Uses the priors for the dirichlet distribution that approximates the true posterior with the optimal
        lower bound, and therefore requires the model to be already trained.


        Parameters
        ----------
        doc_number : int
            Index of the document for which the mixture is returned.

        Returns
        -------
        list of length `self.num_topics`
            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.

        """
    doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]
    return doc_topic[doc_number]