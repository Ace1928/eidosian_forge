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
def test_class_signature():
    info = inspector.info(HasSignature, 'HasSignature')
    assert info['init_definition'] == 'HasSignature(test)'
    assert info['init_docstring'] == HasSignature.__init__.__doc__