from textwrap import dedent
import pytest
from pandas.util._decorators import deprecate
import pandas._testing as tm
def new_func_no_docstring():
    return 'new_func_no_docstring called'