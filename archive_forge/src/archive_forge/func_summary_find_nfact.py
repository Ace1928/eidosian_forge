import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
def summary_find_nfact(self):
    """provides a summary for the selection of the number of factors

        Returns
        -------
        sumstr : str
            summary of the results for selecting the number of factors

        """
    if not hasattr(self, 'results_find_nfact'):
        self.fit_find_nfact()
    results = self.results_find_nfact
    sumstr = ''
    sumstr += '\n' + 'Best result for k, by AIC, BIC, R2_adj, L1O'
    sumstr += '\n' + ' ' * 19 + '%5d %4d %6d %5d' % tuple(self.best_nfact)
    from statsmodels.iolib.table import SimpleTable
    headers = 'k, AIC, BIC, R2_adj, L1O'.split(', ')
    numformat = ['%6d'] + ['%10.3f'] * 4
    txt_fmt1 = dict(data_fmts=numformat)
    tabl = SimpleTable(results, headers, None, txt_fmt=txt_fmt1)
    sumstr += '\n' + 'PCA regression on simulated data,'
    sumstr += '\n' + 'DGP: 2 factors and 4 explanatory variables'
    sumstr += '\n' + tabl.__str__()
    sumstr += '\n' + 'Notes: k is number of components of PCA,'
    sumstr += '\n' + '       constant is added additionally'
    sumstr += '\n' + '       k=0 means regression on constant only'
    sumstr += '\n' + '       L1O: sum of squared prediction errors for leave-one-out'
    return sumstr