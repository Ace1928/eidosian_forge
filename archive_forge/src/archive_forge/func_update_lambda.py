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
def update_lambda(self, sstats, word_list, opt_o):
    """Update appropriate columns of lambda and top level sticks based on documents.

        Parameters
        ----------
        sstats : :class:`~gensim.models.hdpmodel.SuffStats`
            Statistic for all document(s) in the chunk.
        word_list : list of int
            Contains word id of all the unique words in the chunk of documents on which update is being performed.
        opt_o : bool, optional
            If True - invokes a call to :meth:`~gensim.models.hdpmodel.HdpModel.optimal_ordering` to order the topics.

        """
    self.m_status_up_to_date = False
    rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
    if rhot < rhot_bound:
        rhot = rhot_bound
    self.m_rhot = rhot
    self.m_lambda[:, word_list] = self.m_lambda[:, word_list] * (1 - rhot) + rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_chunksize
    self.m_lambda_sum = (1 - rhot) * self.m_lambda_sum + rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_chunksize
    self.m_updatect += 1
    self.m_timestamp[word_list] = self.m_updatect
    self.m_r.append(self.m_r[-1] + np.log(1 - rhot))
    self.m_varphi_ss = (1.0 - rhot) * self.m_varphi_ss + rhot * sstats.m_var_sticks_ss * self.m_D / sstats.m_chunksize
    if opt_o:
        self.optimal_ordering()
    self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T - 1] + 1.0
    var_phi_sum = np.flipud(self.m_varphi_ss[1:])
    self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma