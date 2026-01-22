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
def test_pinfo_docstring_dynamic():
    obj_def = 'class Bar:\n    __custom_documentations__ = {\n     "prop" : "cdoc for prop",\n     "non_exist" : "cdoc for non_exist",\n    }\n    @property\n    def prop(self):\n        \'\'\'\n        Docstring for prop\n        \'\'\'\n        return self._prop\n    \n    @prop.setter\n    def prop(self, v):\n        self._prop = v\n    '
    ip.run_cell(obj_def)
    ip.run_cell('b = Bar()')
    with AssertPrints('Docstring:   cdoc for prop'):
        ip.run_line_magic('pinfo', 'b.prop')
    with AssertPrints('Docstring:   cdoc for non_exist'):
        ip.run_line_magic('pinfo', 'b.non_exist')
    with AssertPrints('Docstring:   cdoc for prop'):
        ip.run_cell('b.prop?')
    with AssertPrints('Docstring:   cdoc for non_exist'):
        ip.run_cell('b.non_exist?')
    with AssertPrints('Docstring:   <no docstring>'):
        ip.run_cell('b.undefined?')