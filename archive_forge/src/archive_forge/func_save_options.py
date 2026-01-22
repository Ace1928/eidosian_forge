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
@deprecated('This method will be removed in 4.0.0, use `save` instead.')
def save_options(self):
    """Writes all the values of the attributes for the current model in "options.dat" file.

        Warnings
        --------
        This method is deprecated, use :meth:`~gensim.models.hdpmodel.HdpModel.save` instead.

        """
    if not self.outputdir:
        logger.error('cannot store options without having specified an output directory')
        return
    fname = '%s/options.dat' % self.outputdir
    with utils.open(fname, 'wb') as fout:
        fout.write('tau: %s\n' % str(self.m_tau - 1))
        fout.write('chunksize: %s\n' % str(self.chunksize))
        fout.write('var_converge: %s\n' % str(self.m_var_converge))
        fout.write('D: %s\n' % str(self.m_D))
        fout.write('K: %s\n' % str(self.m_K))
        fout.write('T: %s\n' % str(self.m_T))
        fout.write('W: %s\n' % str(self.m_W))
        fout.write('alpha: %s\n' % str(self.m_alpha))
        fout.write('kappa: %s\n' % str(self.m_kappa))
        fout.write('eta: %s\n' % str(self.m_eta))
        fout.write('gamma: %s\n' % str(self.m_gamma))