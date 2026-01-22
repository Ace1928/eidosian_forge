from statsmodels.compat.pandas import assert_frame_equal
from numpy.testing import (
import pandas as pd
from statsmodels.stats.anova import AnovaRM
def test_two_factors_repeated_measures_anova():
    """
    Testing two factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:48, :], 'DV', 'id', within=['A', 'B']).fit()
    a = [[1, 7, 40.14159, 0.0003905263], [2, 14, 29.21739, 1.007549e-05], [2, 14, 17.10545, 0.0001741322]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)