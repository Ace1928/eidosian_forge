import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_current_backend_plot_and_style(self):
    obj = ExampleElement([]).options(style_opt1='A', plot_opt1='B')
    plot_opts = Store.lookup_options('backend_1', obj, 'plot')
    assert plot_opts.options == {'plot_opt1': 'B'}
    style_opts = Store.lookup_options('backend_1', obj, 'style')
    assert style_opts.options == {'style_opt1': 'A'}