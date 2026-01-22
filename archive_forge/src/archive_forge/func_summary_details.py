from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def summary_details(self, source='news', impact_date=None, impacted_variable=None, update_date=None, updated_variable=None, groupby='update date', sparsify=True, float_format='%.2f', multiple_tables=False):
    """
        Create summary table with detailed impacts; by date, variable

        Parameters
        ----------
        source : {news, revisions}
            The source of impacts to summarize. Default is "news".
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
        update_date : int, str, datetime, list, array, or slice, optional
            Observation index label or slice of labels specifying particular
            updated periods to display. The updated date(s) describe the
            periods in which the new data points were available that generated
            the news). See the note on `impact_date` for details about what
            these labels are.
        updated_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying particular
            updated variables to display. The updated variable(s) describe the
            variables that were *affected* by the news. If you do not know the
            labels for the variables, check the `endog_names` attribute of the
            model instance.
        groupby : {update date, updated date, impact date, impacted date}
            The primary variable for grouping results in the details table. The
            default is to group by update date.
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.
        float_format : str, optional
            Formatter format string syntax for converting numbers to strings.
            Default is '%.2f'.
        multiple_tables : bool, optional
            If set to True, this function will return a list of tables, one
            table for each of the unique `groupby` levels. Default is False,
            in which case this function returns a single table.

        Returns
        -------
        details_table : SimpleTable or list of SimpleTable
            Table or list of tables describing how the news from each update
            (i.e. news from a particular variable / date) translates into
            changes to the forecasts of each impacted variable variable / date.

            This table contains information about the updates and about the
            impacts. Updates are newly observed datapoints that were not
            available in the previous results set. Each update leads to news,
            and the news may cause changes in the forecasts of the impacted
            variables. The amount that a particular piece of news (from an
            update to some variable at some date) impacts a variable at some
            date depends on weights that can be computed from the model
            results.

            The data contained in this table that refer to updates are:

            - `update date` : The date at which a new datapoint was added.
            - `updated variable` : The variable for which a new datapoint was
              added.
            - `forecast (prev)` : The value that had been forecast by the
              previous model for the given updated variable and date.
            - `observed` : The observed value of the new datapoint.
            - `news` : The news is the difference between the observed value
              and the previously forecast value for a given updated variable
              and date.

            The data contained in this table that refer to impacts are:

            - `impact date` : A date associated with an impact.
            - `impacted variable` : A variable that was impacted by the news.
            - `weight` : The weight of news from a given `update date` and
              `update variable` on a given `impacted variable` at a given
              `impact date`.
            - `impact` : The revision to the smoothed estimate / forecast of
              the impacted variable at the impact date based specifically on
              the news generated by the `updated variable` at the
              `update date`.

        See Also
        --------
        details_by_impact
        details_by_update
        """
    if self.k_endog == 1:
        if impacted_variable is None:
            impacted_variable = self.endog_names[0]
        if updated_variable is None:
            updated_variable = self.endog_names[0]
    s = list(np.s_[:, :, :, :, :, :])
    if impact_date is not None:
        s[0] = np.s_[impact_date]
    if impacted_variable is not None:
        s[1] = np.s_[impacted_variable]
    if update_date is not None:
        s[2] = np.s_[update_date]
    if updated_variable is not None:
        s[3] = np.s_[updated_variable]
    s = tuple(s)
    if source == 'news':
        details = self.details_by_impact.loc[s, :]
        columns = {'current': 'observed', 'prev': 'forecast (prev)', 'update date': 'update date', 'updated variable': 'updated variable', 'news': 'news'}
    elif source == 'revisions':
        details = self.revision_details_by_impact.loc[s, :]
        columns = {'current': 'revised', 'prev': 'observed (prev)', 'update date': 'revision date', 'updated variable': 'revised variable', 'news': 'revision'}
    else:
        raise ValueError(f'Invalid `source`: {source}. Must be "news" or "revisions".')
    groupby = groupby.lower().replace('_', ' ')
    groupby_overall = 'impact'
    levels_order = [0, 1, 2, 3]
    if groupby == 'update date':
        levels_order = [2, 3, 0, 1]
        groupby_overall = 'update'
    elif groupby == 'updated variable':
        levels_order = [3, 2, 1, 0]
        groupby_overall = 'update'
    elif groupby == 'impacted variable':
        levels_order = [1, 0, 3, 2]
    elif groupby != 'impact date':
        raise ValueError(f'Invalid groupby for details table. Valid options are "update date", "updated variable", "impact date",or "impacted variable". Got "{groupby}".')
    details.index = details.index.reorder_levels(levels_order).remove_unused_levels()
    details = details.sort_index()
    base_levels = [0, 1, 2, 3]
    if groupby_overall == 'update':
        details.set_index([columns['current'], columns['prev']], append=True, inplace=True)
        details.index = details.index.reorder_levels([0, 1, 4, 5, 2, 3])
        base_levels = [0, 1, 4, 5]
    tmp_index = details.index.remove_unused_levels()
    n_levels = len(tmp_index.levels)
    k_level_values = [len(tmp_index.levels[i]) for i in range(n_levels)]
    removed_levels = []
    if sparsify:
        for i in sorted(base_levels)[::-1][:-1]:
            if k_level_values[i] == 1:
                name = tmp_index.names[i]
                value = tmp_index.levels[i][0]
                can_drop = name == columns['update date'] and update_date is not None or (name == columns['updated variable'] and updated_variable is not None) or (name == 'impact date' and impact_date is not None) or (name == 'impacted variable' and (impacted_variable is not None or self.impacted_variable is not None))
                if can_drop or not multiple_tables:
                    removed_levels.insert(0, f'{name} = {value}')
                    details.index = tmp_index = tmp_index.droplevel(i)
    details = details.reset_index()

    def str_format(num, mark_ones=False, mark_zeroes=False):
        if pd.isnull(num):
            out = ''
        elif mark_ones and np.abs(1 - num) < self.tolerance:
            out = '1.0'
        elif mark_zeroes and np.abs(num) < self.tolerance:
            out = '0'
        else:
            out = float_format % num
        return out

    def create_table(details, removed_levels):
        for key in [columns['current'], columns['prev'], columns['news'], 'weight', 'impact']:
            if key in details:
                args = (True if key in ['weight'] else False, True if key in ['weight', 'impact'] else False)
                details[key] = details[key].apply(str_format, args=args)
        for key in [columns['update date'], 'impact date']:
            if key in details:
                details[key] = details[key].apply(str)
        if sparsify:
            sparsify_cols = [columns['update date'], columns['updated variable'], 'impact date', 'impacted variable']
            data_cols = [columns['current'], columns['prev']]
            if groupby_overall == 'update':
                sparsify_cols = data_cols + sparsify_cols
            for key in sparsify_cols:
                if key in details:
                    mask = details[key] == details[key].shift(1)
                    if key in data_cols:
                        if columns['update date'] in details:
                            tmp = details[columns['update date']]
                            mask &= tmp == tmp.shift(1)
                        if columns['updated variable'] in details:
                            tmp = details[columns['updated variable']]
                            mask &= tmp == tmp.shift(1)
                    details.loc[mask, key] = ''
        params_data = details.values
        params_header = [str(x) for x in details.columns.tolist()]
        params_stubs = None
        title = f'Details of {source}'
        if len(removed_levels):
            title += ' for [' + ', '.join(removed_levels) + ']'
        return SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
    if multiple_tables:
        details_table = []
        for item in details[columns[groupby]].unique():
            mask = details[columns[groupby]] == item
            item_details = details[mask].drop(columns[groupby], axis=1)
            item_removed_levels = [f'{columns[groupby]} = {item}'] + removed_levels
            details_table.append(create_table(item_details, item_removed_levels))
    else:
        details_table = create_table(details, removed_levels)
    return details_table