import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_clone(self):
    vals = np.random.rand(20, 20)
    xs = np.linspace(0, 10, 20)
    ys = np.linspace(0, 10, 20)
    ys[-1] += 0.001
    img = Image({'vals': vals, 'xs': xs, 'ys': ys}, ['xs', 'ys'], 'vals', rtol=0.01)
    self.assertEqual(img.clone().rtol, 0.01)