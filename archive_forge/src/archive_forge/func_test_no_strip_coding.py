import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_no_strip_coding(self):
    src = '\n'.join(['%%writefile foo.py', '# coding: utf-8', 'print(u"üñîçø∂é")'])
    out = self.sp.transform_cell(src)
    assert '# coding: utf-8' in out