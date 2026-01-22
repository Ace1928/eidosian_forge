import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_opts_resassignment_cleans_unused_tree(self):
    obj = ExampleElement([]).opts(style_opt1='A').opts(plot_opt1='B')
    custom_options = Store._custom_options['backend_1']
    self.assertIn(obj.id, custom_options)
    self.assertEqual(len(custom_options), 1)