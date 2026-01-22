from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
def test_pandas_dependency_message(fetch_kddcup99_fxt, hide_available_pandas):
    check_pandas_dependency_message(fetch_kddcup99_fxt)