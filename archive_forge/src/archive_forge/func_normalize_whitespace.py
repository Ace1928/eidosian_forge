from numpy.distutils.from_template import process_str
from numpy.testing import assert_equal
def normalize_whitespace(s):
    """
    Remove leading and trailing whitespace, and convert internal
    stretches of whitespace to a single space.
    """
    return ' '.join(s.split())