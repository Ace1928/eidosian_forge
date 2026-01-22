import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
@property
def col_names(self):
    """column names for summary table
        """
    pr_test = 'P>%s' % self.distribution
    col_names = [self.distribution, pr_test, 'df constraint']
    if self.distribution == 'F':
        col_names.append('df denom')
    return col_names