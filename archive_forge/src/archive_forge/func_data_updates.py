from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def data_updates(self):
    """
        Updated data; new entries that did not exist in the previous dataset

        Returns
        -------
        data_updates : pd.DataFrame
            Index is as MultiIndex consisting of `update date` and
            `updated variable`. The columns are:

            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
              (recall that for these updated data points, the previous dataset
              had no observed value for them at all)
            - `observed`: the value of the new entry, as it is observed in the
              new dataset

        See also
        --------
        data_revisions
        """
    data = pd.concat([self.update_realized.rename('observed'), self.update_forecasts.rename('forecast (prev)')], axis=1).sort_index()
    return data