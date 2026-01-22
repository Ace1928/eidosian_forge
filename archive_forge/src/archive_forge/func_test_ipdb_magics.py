import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
def test_ipdb_magics():
    '''Test calling some IPython magics from ipdb.

    First, set up some test functions and classes which we can inspect.

    >>> class ExampleClass(object):
    ...    """Docstring for ExampleClass."""
    ...    def __init__(self):
    ...        """Docstring for ExampleClass.__init__"""
    ...        pass
    ...    def __str__(self):
    ...        return "ExampleClass()"

    >>> def example_function(x, y, z="hello"):
    ...     """Docstring for example_function."""
    ...     pass

    >>> old_trace = sys.gettrace()

    Create a function which triggers ipdb.

    >>> def trigger_ipdb():
    ...    a = ExampleClass()
    ...    debugger.Pdb().set_trace()

    >>> with PdbTestInput([
    ...    'pdef example_function',
    ...    'pdoc ExampleClass',
    ...    'up',
    ...    'down',
    ...    'list',
    ...    'pinfo a',
    ...    'll',
    ...    'continue',
    ... ]):
    ...     trigger_ipdb()
    --Return--
    None
    > <doctest ...>(3)trigger_ipdb()
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> pdef example_function
     example_function(x, y, z='hello')
     ipdb> pdoc ExampleClass
    Class docstring:
        Docstring for ExampleClass.
    Init docstring:
        Docstring for ExampleClass.__init__
    ipdb> up
    > <doctest ...>(11)<module>()
          7    'pinfo a',
          8    'll',
          9    'continue',
         10 ]):
    ---> 11     trigger_ipdb()
    <BLANKLINE>
    ipdb> down
    None
    > <doctest ...>(3)trigger_ipdb()
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> list
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> pinfo a
    Type:           ExampleClass
    String form:    ExampleClass()
    Namespace:      Local...
    Docstring:      Docstring for ExampleClass.
    Init docstring: Docstring for ExampleClass.__init__
    ipdb> ll
          1 def trigger_ipdb():
          2    a = ExampleClass()
    ----> 3    debugger.Pdb().set_trace()
    <BLANKLINE>
    ipdb> continue
    
    Restore previous trace function, e.g. for coverage.py    
    
    >>> sys.settrace(old_trace)
    '''