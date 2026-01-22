from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def impacts(self):
    """
        Impacts from news and revisions on all dates / variables of interest

        Returns
        -------
        impacts : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `estimate (prev)`: the previous estimate / forecast of the
              date / variable of interest.
            - `impact of revisions`: the impact of all data revisions on
              the estimate of the date / variable of interest.
            - `impact of news`: the impact of all news on the estimate of
              the date / variable of interest.
            - `total impact`: the total impact of both revisions and news on
              the estimate of the date / variable of interest.
            - `estimate (new)`: the new estimate / forecast of the
              date / variable of interest after taking into account the effects
              of the revisions and news.

        Notes
        -----
        This table decomposes updated forecasts of variables of interest into
        the overall effect from revisions and news.

        This table does not break down the detail by the updated
        dates / variables. That information can be found in the
        `details_by_impact` `details_by_update` tables.

        See Also
        --------
        details_by_impact
        details_by_update
        """
    impacts = pd.concat([self.prev_impacted_forecasts.unstack().rename('estimate (prev)'), self.revision_impacts.unstack().rename('impact of revisions'), self.update_impacts.unstack().rename('impact of news'), self.post_impacted_forecasts.unstack().rename('estimate (new)')], axis=1)
    impacts['impact of revisions'] = impacts['impact of revisions'].astype(float).fillna(0)
    impacts['impact of news'] = impacts['impact of news'].astype(float).fillna(0)
    impacts['total impact'] = impacts['impact of revisions'] + impacts['impact of news']
    impacts = impacts.reorder_levels([1, 0]).sort_index()
    impacts.index.names = ['impact date', 'impacted variable']
    impacts = impacts[['estimate (prev)', 'impact of revisions', 'impact of news', 'total impact', 'estimate (new)']]
    if self.impacted_variable is not None:
        impacts = impacts.loc[np.s_[:, self.impacted_variable], :]
    tmp = np.abs(impacts[['impact of revisions', 'impact of news']])
    mask = (tmp > self.tolerance).any(axis=1)
    return impacts[mask]