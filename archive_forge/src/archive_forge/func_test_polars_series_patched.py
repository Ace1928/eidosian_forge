from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_polars_series_patched(self):
    import polars as pl
    pseries = pl.Series([0, 1, 2])
    self.assertIsInstance(pseries.hvplot, hvPlotTabular)