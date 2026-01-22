from textwrap import dedent
import pytest
from pandas.util._decorators import deprecate
import pandas._testing as tm
def new_func_wrong_docstring():
    """Summary should be in the next line."""
    return 'new_func_wrong_docstring called'