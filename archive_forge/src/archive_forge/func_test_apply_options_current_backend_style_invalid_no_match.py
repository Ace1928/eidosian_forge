import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
def test_apply_options_current_backend_style_invalid_no_match(self):
    err = "Unexpected option 'zxy' for ExampleElement type across all extensions\\. No similar options found\\."
    with self.assertRaisesRegex(ValueError, err):
        ExampleElement([]).options(zxy='A')