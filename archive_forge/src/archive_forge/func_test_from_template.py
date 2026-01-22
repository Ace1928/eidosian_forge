from numpy.distutils.from_template import process_str
from numpy.testing import assert_equal
def test_from_template():
    """Regression test for gh-10712."""
    pyf = process_str(pyf_src)
    normalized_pyf = normalize_whitespace(pyf)
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    assert_equal(normalized_pyf, normalized_expected_pyf)