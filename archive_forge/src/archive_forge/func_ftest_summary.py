import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
def ftest_summary(self):
    """run all ftests on the joint model

        Returns
        -------
        fres : str
           a string that lists the results of all individual f-tests
        summarytable : list of tuples
           contains (pair, (fvalue, pvalue,df_denom, df_num)) for each f-test

        Note
        ----
        This are the raw results and not formatted for nice printing.

        """
    if not hasattr(self, 'lsjoint'):
        self.fitjoint()
    txt = []
    summarytable = []
    txt.append('F-test for equality of coefficients across groups')
    fres = self.lsjoint.f_test(self.contrasts['all'])
    txt.append(fres.__str__())
    summarytable.append(('all', (fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
    pairs = np.triu_indices(len(self.unique), 1)
    for ind1, ind2 in zip(*pairs):
        g1 = self.unique[ind1]
        g2 = self.unique[ind2]
        txt.append('F-test for equality of coefficients between group %s and group %s' % (g1, g2))
        group = (g1, g2)
        fres = self.lsjoint.f_test(self.contrasts[group])
        txt.append(fres.__str__())
        summarytable.append((group, (fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
    self.summarytable = summarytable
    return ('\n'.join(txt), summarytable)