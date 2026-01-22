import textwrap
from io import BytesIO
import pytest
from sklearn.datasets._arff_parser import (
def test_load_arff_from_gzip_file_error_parser():
    """An error will be raised if the parser is not known."""
    err_msg = "Unknown parser: 'xxx'. Should be 'liac-arff' or 'pandas'"
    with pytest.raises(ValueError, match=err_msg):
        load_arff_from_gzip_file('xxx', 'xxx', 'xxx', 'xxx', 'xxx', 'xxx')