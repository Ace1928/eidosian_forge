import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def ddt(arg=None, **kwargs):
    """
    Class decorator for subclasses of ``unittest.TestCase``.

    Apply this decorator to the test case class, and then
    decorate test methods with ``@data``.

    For each method decorated with ``@data``, this will effectively create as
    many methods as data items are passed as parameters to ``@data``.

    The names of the test methods follow the pattern
    ``original_test_name_{ordinal}_{data}``. ``ordinal`` is the position of the
    data argument, starting with 1.

    For data we use a string representation of the data value converted into a
    valid python identifier.  If ``data.__name__`` exists, we use that instead.

    For each method decorated with ``@file_data('test_data.json')``, the
    decorator will try to load the test_data.json file located relative
    to the python file containing the method that is decorated. It will,
    for each ``test_name`` key create as many methods in the list of values
    from the ``data`` key.

    Decorating with the keyword argument ``testNameFormat`` can control the
    format of the generated test names.  For example:

    - ``@ddt(testNameFormat=TestNameFormat.DEFAULT)`` will be index and values.

    - ``@ddt(testNameFormat=TestNameFormat.INDEX_ONLY)`` will be index only.

    - ``@ddt`` is the same as DEFAULT.

    """
    fmt_test_name = kwargs.get('testNameFormat', TestNameFormat.DEFAULT)

    def wrapper(cls):
        for name, func in list(cls.__dict__.items()):
            if hasattr(func, DATA_ATTR):
                index_len = getattr(func, INDEX_LEN)
                for i, v in enumerate(getattr(func, DATA_ATTR)):
                    test_name = mk_test_name(name, getattr(v, '__name__', v), i, index_len, fmt_test_name)
                    test_data_docstring = _get_test_data_docstring(func, v)
                    if hasattr(func, UNPACK_ATTR):
                        if isinstance(v, tuple) or isinstance(v, list):
                            add_test(cls, test_name, test_data_docstring, func, *v)
                        else:
                            add_test(cls, test_name, test_data_docstring, func, **v)
                    else:
                        add_test(cls, test_name, test_data_docstring, func, v)
                delattr(cls, name)
            elif hasattr(func, FILE_ATTR):
                file_attr = getattr(func, FILE_ATTR)
                process_file_data(cls, name, func, file_attr)
                delattr(cls, name)
        return cls
    return wrapper(arg) if inspect.isclass(arg) else wrapper