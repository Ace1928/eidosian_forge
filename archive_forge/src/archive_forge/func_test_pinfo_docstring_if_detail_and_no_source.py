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
def test_pinfo_docstring_if_detail_and_no_source():
    """ Docstring should be displayed if source info not available """
    obj_def = 'class Foo(object):\n                  """ This is a docstring for Foo """\n                  def bar(self):\n                      """ This is a docstring for Foo.bar """\n                      pass\n              '
    ip.run_cell(obj_def)
    ip.run_cell('foo = Foo()')
    with AssertNotPrints('Source:'):
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=0)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo', detail_level=1)
        with AssertPrints('Docstring:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=0)
    with AssertNotPrints('Docstring:'):
        with AssertPrints('Source:'):
            ip._inspect('pinfo', 'foo.bar', detail_level=1)