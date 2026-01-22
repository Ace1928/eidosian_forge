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
def test_calldef_none():
    for obj in [support_function_one, SimpleClass().method, any, str.upper]:
        i = inspector.info(obj)
        assert i['call_def'] is None