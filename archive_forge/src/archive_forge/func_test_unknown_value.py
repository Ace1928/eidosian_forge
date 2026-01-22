from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
def test_unknown_value(self):
    """
        Unknown eol values should raise an error.
        """
    prefs = (('eol', 'unknown-value'),)
    self.assertRaises(errors.BzrError, _get_filter_stack_for, prefs)