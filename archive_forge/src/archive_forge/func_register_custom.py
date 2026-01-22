import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
@classmethod
def register_custom(cls, objtype, backend, custom_plot=None, custom_style=None):
    if custom_style is None:
        custom_style = []
    if custom_plot is None:
        custom_plot = []
    groups = Options._option_groups
    if backend not in Store._options:
        Store._options[backend] = OptionTree([], groups=groups)
        Store._custom_options[backend] = {}
    name = objtype.__name__
    style_opts = Keywords(['style_opt1', 'style_opt2'] + custom_style, name)
    plot_opts = Keywords(['plot_opt1', 'plot_opt2'] + custom_plot, name)
    opt_groups = {'plot': Options(allowed_keywords=plot_opts), 'style': Options(allowed_keywords=style_opts), 'output': Options(allowed_keywords=['backend'])}
    Store._options[backend][name] = opt_groups
    Store.renderers[backend] = MockRenderer(backend)