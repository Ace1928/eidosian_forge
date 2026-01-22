from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def get_impacts(self, groupby=None, include_revisions=True, include_updates=True):
    details = self.get_details(include_revisions=include_revisions, include_updates=include_updates)
    impacts = details['impact'].unstack(['impact date', 'impacted variable'])
    if groupby is not None:
        impacts = impacts.unstack('update date').groupby(groupby).sum(min_count=1).stack('update date').swaplevel().sort_index()
    return impacts