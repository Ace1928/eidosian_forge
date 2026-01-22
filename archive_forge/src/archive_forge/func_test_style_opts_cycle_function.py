from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_cycle_function(self):
    import numpy as np
    np.random.seed(42)
    line = 'Curve (color=Cycle(values=list(np.random.rand(3,3))))'
    options = OptsSpec.parse(line, {'np': np, 'Cycle': Cycle})
    self.assertTrue('Curve' in options)
    self.assertTrue('style' in options['Curve'])
    self.assertTrue('color' in options['Curve']['style'].kwargs)
    self.assertTrue(isinstance(options['Curve']['style'].kwargs['color'], Cycle))
    values = np.array([[0.37454012, 0.95071431, 0.73199394], [0.59865848, 0.15601864, 0.15599452], [0.05808361, 0.86617615, 0.60111501]])
    self.assertEqual(np.array(options['Curve']['style'].kwargs['color'].values), values)