import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_current_backend_style_invalid(self):
    err = "Unexpected option 'style_opt3' for ExampleElement type across all extensions. Similar options for current extension \\('backend_1'\\) are: \\['style_opt1', 'style_opt2'\\]\\."
    with self.assertRaisesRegex(ValueError, err):
        ExampleElement([]).options(style_opt3='A')