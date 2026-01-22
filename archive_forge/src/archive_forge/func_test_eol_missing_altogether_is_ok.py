from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
def test_eol_missing_altogether_is_ok(self):
    """
        Not having eol in the set of preferences should be ok.
        """
    prefs = (('eol', None),)
    self.assertEqual([], _get_filter_stack_for(prefs))