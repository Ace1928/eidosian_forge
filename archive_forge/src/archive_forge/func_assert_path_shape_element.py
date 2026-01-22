import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def assert_path_shape_element(self, shape, element, xref='x', yref='y'):
    self.assertEqual(shape['type'], 'path')
    expected_path = 'M' + 'L'.join([f'{x} {y}' for x, y in zip(element.dimension_values(0), element.dimension_values(1))]) + 'Z'
    self.assertEqual(shape['path'], expected_path)
    self.assertEqual(shape['xref'], xref)
    self.assertEqual(shape['yref'], yref)