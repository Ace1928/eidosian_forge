from unittest import SkipTest
from pyviz_comms import CommManager
from holoviews import Store, notebook_extension
from holoviews.core.options import OptionTree
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import bokeh, mpl
from holoviews.util import Options, OutputSettings, opts, output
from ..utils import LoggingComparisonTestCase
def test_output_util_png_kwargs(self):
    self.assertEqual(OutputSettings.options.get('fig', None), None)
    output(fig='png')
    self.assertEqual(OutputSettings.options.get('fig', None), 'png')