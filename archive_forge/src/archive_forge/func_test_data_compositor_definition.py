from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_data_compositor_definition(self):
    definition = ' data transform(Image * Image) HCSTEST'
    self.line_magic('compositor', definition)
    compositors = [c for c in Compositor.definitions if c.group == 'HCSTEST']
    self.assertEqual(len(compositors), 1)
    self.assertEqual(compositors[0].group, 'HCSTEST')
    self.assertEqual(compositors[0].mode, 'data')