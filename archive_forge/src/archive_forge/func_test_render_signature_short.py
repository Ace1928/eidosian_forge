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
def test_render_signature_short():

    def short_fun(a=1):
        pass
    sig = oinspect._render_signature(signature(short_fun), short_fun.__name__)
    assert sig == 'short_fun(a=1)'