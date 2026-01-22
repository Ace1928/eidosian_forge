from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
def update_chunk(self, chunk, update=True, opt_o=True):
    """Performs lazy update on necessary columns of lambda and variational inference for documents in the chunk.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        update : bool, optional
            If True - call :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda`.
        opt_o : bool, optional
            Passed as argument to :meth:`~gensim.models.hdpmodel.HdpModel.update_lambda`.
            If True then the topics will be ordered, False otherwise.

        Returns
        -------
        (float, int)
            A tuple of likelihood and sum of all the word counts from each document in the corpus.

        """
    unique_words = dict()
    word_list = []
    for doc in chunk:
        for word_id, _ in doc:
            if word_id not in unique_words:
                unique_words[word_id] = len(unique_words)
                word_list.append(word_id)
    wt = len(word_list)
    rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
    self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
    self.m_Elogbeta[:, word_list] = psi(self.m_eta + self.m_lambda[:, word_list]) - psi(self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])
    ss = SuffStats(self.m_T, wt, len(chunk))
    Elogsticks_1st = expect_log_sticks(self.m_var_sticks)
    score = 0.0
    count = 0
    for doc in chunk:
        if len(doc) > 0:
            doc_word_ids, doc_word_counts = zip(*doc)
            doc_score = self.doc_e_step(ss, Elogsticks_1st, unique_words, doc_word_ids, doc_word_counts, self.m_var_converge)
            count += sum(doc_word_counts)
            score += doc_score
    if update:
        self.update_lambda(ss, word_list, opt_o)
    return (score, count)