import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
def skip_on_raise(*exc_types):
    """
    Decorate a test function to translation specific exception types to :exc:`unittest.SkipTest`.

    :param exc_types: One or more positional arguments give the exception
                      types to be translated to :exc:`unittest.SkipTest`.
    :returns: A decorator function specialized to `exc_types`.
    """

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kw):
            try:
                return function(*args, **kw)
            except exc_types as e:
                logger.debug('Translating exception to unittest.SkipTest ..', exc_info=True)
                raise unittest.SkipTest('skipping test because %s was raised' % type(e))
        return wrapper
    return decorator