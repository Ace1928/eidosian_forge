from unittest import SkipTest
from pyviz_comms import CommManager
from holoviews import Store, notebook_extension
from holoviews.core.options import OptionTree
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import bokeh, mpl
from holoviews.util import Options, OutputSettings, opts, output
from ..utils import LoggingComparisonTestCase
def test_opts_builder_repr_line_magic(self):
    magic = "%opts Bivariate [bandwidth=0.5] (cmap='jet') Points [logx=True] (size=2)"
    expected = ["opts.Bivariate(bandwidth=0.5, cmap='jet')", 'opts.Points(logx=True, size=2)']
    reprs = opts._builder_reprs(magic)
    self.assertEqual(reprs, expected)