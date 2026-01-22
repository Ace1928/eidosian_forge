from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_pinfo_bool_raise():
    """
    Test that bool method is not called on parent.
    """

    class RaiseBool:
        attr = None

        def __bool__(self):
            raise ValueError('pinfo should not access this method')
    raise_bool = RaiseBool()
    with cleanup_user_ns(raise_bool=raise_bool):
        ip._inspect('pinfo', 'raise_bool.attr', detail_level=0)