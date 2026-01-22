from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def summary_impacts(self, impact_date=None, impacted_variable=None, groupby='impact date', show_revisions_columns=None, sparsify=True, float_format='%.2f'):
    """
        Create summary table with detailed impacts from news; by date, variable

        Parameters
        ----------
        impact_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            impact periods to display. The impact date(s) describe the periods
            in which impacted variables were *affected* by the news. If this
            argument is given, the output table will only show this impact date
            or dates. Note that this argument is passed to the Pandas `loc`
            accessor, and so it should correspond to the labels of the model's
            index. If the model was created with data in a list or numpy array,
            then these labels will be zero-indexes observation integers.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            impacted variables to display. The impacted variable(s) describe
            the variables that were *affected* by the news. If you do not know
            the labels for the variables, check the `endog_names` attribute of
            the model instance.
        groupby : {impact date, impacted date}
            The primary variable for grouping results in the impacts table. The
            default is to group by update date.
        show_revisions_columns : bool, optional
            If set to False, the impacts table will not show the impacts from
            data revisions or the total impacts. Default is to show the
            revisions and totals columns if any revisions were made and
            otherwise to hide them.
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.
        float_format : str, optional
            Formatter format string syntax for converting numbers to strings.
            Default is '%.2f'.

        Returns
        -------
        impacts_table : SimpleTable
            Table describing total impacts from both revisions and news. See
            the documentation for the `impacts` attribute for more details
            about the index and columns.

        See Also
        --------
        impacts
        """
    if impacted_variable is None and self.k_endog == 1:
        impacted_variable = self.endog_names[0]
    if show_revisions_columns is None:
        show_revisions_columns = self.n_revisions > 0
    s = list(np.s_[:, :])
    if impact_date is not None:
        s[0] = np.s_[impact_date]
    if impacted_variable is not None:
        s[1] = np.s_[impacted_variable]
    s = tuple(s)
    impacts = self.impacts.loc[s, :]
    groupby = groupby.lower()
    if groupby in ['impacted variable', 'impacted_variable']:
        impacts.index = impacts.index.swaplevel(1, 0)
    elif groupby not in ['impact date', 'impact_date']:
        raise ValueError(f'Invalid groupby for impacts table. Valid options are "impact date" or "impacted variable".Got "{groupby}".')
    impacts = impacts.sort_index()
    tmp_index = impacts.index.remove_unused_levels()
    k_vars = len(tmp_index.levels[1])
    removed_level = None
    if sparsify and k_vars == 1:
        name = tmp_index.names[1]
        value = tmp_index.levels[1][0]
        removed_level = f'{name} = {value}'
        impacts.index = tmp_index.droplevel(1)
        try:
            impacts = impacts.applymap(lambda num: '' if pd.isnull(num) else float_format % num)
        except AttributeError:
            impacts = impacts.map(lambda num: '' if pd.isnull(num) else float_format % num)
        impacts = impacts.reset_index()
        impacts.iloc[:, 0] = impacts.iloc[:, 0].map(str)
    else:
        impacts = impacts.reset_index()
        try:
            impacts.iloc[:, :2] = impacts.iloc[:, :2].applymap(str)
            impacts.iloc[:, 2:] = impacts.iloc[:, 2:].applymap(lambda num: '' if pd.isnull(num) else float_format % num)
        except AttributeError:
            cols = impacts.columns[:2]
            impacts[cols] = impacts[cols].map(str)
            cols = impacts.columns[2:]
            impacts[cols] = impacts[cols].map(lambda num: '' if pd.isnull(num) else float_format % num)
    if sparsify and groupby in impacts:
        mask = impacts[groupby] == impacts[groupby].shift(1)
        tmp = impacts.loc[mask, groupby]
        if len(tmp) > 0:
            impacts.loc[mask, groupby] = ''
    if not show_revisions_columns:
        impacts.drop(['impact of revisions', 'total impact'], axis=1, inplace=True)
    params_data = impacts.values
    params_header = impacts.columns.tolist()
    params_stubs = None
    title = 'Impacts'
    if removed_level is not None:
        join = 'on' if groupby == 'date' else 'for'
        title += f' {join} [{removed_level}]'
    impacts_table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
    return impacts_table