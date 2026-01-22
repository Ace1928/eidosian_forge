from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_output_invalid_size(self):
    self.line_magic('output', 'size=-50')
    self.assertEqual(hv.util.OutputSettings.options.get('size', None), None)