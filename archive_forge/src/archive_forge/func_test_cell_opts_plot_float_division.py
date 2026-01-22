from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
def test_cell_opts_plot_float_division(self):
    self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")
    self.assertEqual(self.get_object('mat1').id, None)
    self.cell_magic('opts', ' Image [aspect=3/4]', 'mat1')
    self.assertEqual(self.get_object('mat1').id, 0)
    assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
    self.assertEqual(Store.lookup_options('matplotlib', self.get_object('mat1'), 'plot').options.get('aspect', False), 3 / 4.0)