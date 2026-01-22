import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script_without_output
Tests for making sure experimental imports work as expected.