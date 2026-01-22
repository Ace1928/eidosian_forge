from textwrap import dedent
import pytest
from pandas.util._decorators import deprecate
import pandas._testing as tm
def new_func_with_deprecation():
    """
    This is the summary. The deprecate directive goes next.

    .. deprecated:: 1.0
        Use new_func instead.

    This is the extended summary. The deprecate directive goes before this.
    """