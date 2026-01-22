import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def mk_test_name(name, value, index=0, index_len=5, name_fmt=TestNameFormat.DEFAULT):
    """
    Generate a new name for a test case.

    It will take the original test name and append an ordinal index and a
    string representation of the value, and convert the result into a valid
    python identifier by replacing extraneous characters with ``_``.

    We avoid doing str(value) if dealing with non-trivial values.
    The problem is possible different names with different runs, e.g.
    different order of dictionary keys (see PYTHONHASHSEED) or dealing
    with mock objects.
    Trivial scalar values are passed as is.

    A "trivial" value is a plain scalar, or a tuple or list consisting
    only of trivial values.

    The test name format is controlled by enum ``TestNameFormat`` as well. See
    the enum documentation for further details.
    """
    index = '{0:0{1}}'.format(index + 1, index_len)
    if name_fmt is TestNameFormat.INDEX_ONLY or not is_trivial(value):
        return '{0}_{1}'.format(name, index)
    try:
        value = str(value)
    except UnicodeEncodeError:
        value = value.encode('ascii', 'backslashreplace')
    test_name = '{0}_{1}_{2}'.format(name, index, value)
    return re.sub('\\W|^(?=\\d)', '_', test_name)