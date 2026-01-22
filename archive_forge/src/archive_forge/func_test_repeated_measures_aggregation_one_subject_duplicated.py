from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_repeated_measures_aggregation_one_subject_duplicated():
    df1 = AnovaRM(data, 'DV', 'id', within=['A', 'B', 'D']).fit()
    data2 = pd.concat([data, data.loc[data['id'] == '1', :]], axis=0)
    data2 = data2.reset_index()
    df2 = AnovaRM(data2, 'DV', 'id', within=['A', 'B', 'D'], aggregate_func=pd.Series.mean).fit()
    assert_frame_equal(df1.anova_table, df2.anova_table)