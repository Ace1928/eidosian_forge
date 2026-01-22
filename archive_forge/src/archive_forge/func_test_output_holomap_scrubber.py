from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_output_holomap_scrubber(self):
    self.line_magic('output', "holomap='scrubber'")
    self.assertEqual(hv.util.OutputSettings.options.get('holomap', None), 'scrubber')