from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
def test_other_known_values(self):
    """These known eol values have corresponding filters."""
    known_values = ('lf', 'crlf', 'native', 'native-with-crlf-in-repo', 'lf-with-crlf-in-repo', 'crlf-with-crlf-in-repo')
    for value in known_values:
        prefs = (('eol', value),)
        self.assertNotEqual([], _get_filter_stack_for(prefs))