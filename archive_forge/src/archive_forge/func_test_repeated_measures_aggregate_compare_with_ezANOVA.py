from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_repeated_measures_aggregate_compare_with_ezANOVA():
    ez = pd.DataFrame({'F Value': [8.7650709, 8.4985785, 20.5076546, 0.8457797, 21.7593382, 6.2416695, 5.4253359], 'Num DF': [1, 2, 1, 2, 1, 2, 2], 'Den DF': [7, 14, 7, 14, 7, 14, 14], 'Pr > F': [0.021087505, 0.003833921, 0.002704428, 0.450021759, 0.002301792, 0.011536846, 0.018010647]}, index=pd.Index(['A', 'B', 'D', 'A:B', 'A:D', 'B:D', 'A:B:D']))
    ez = ez[['F Value', 'Num DF', 'Den DF', 'Pr > F']]
    double_data = pd.concat([data, data], axis=0)
    df = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'], aggregate_func=pd.Series.mean).fit().anova_table
    assert_frame_equal(ez, df, check_dtype=False)