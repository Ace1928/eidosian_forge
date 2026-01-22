import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_whole_cell(self):
    src = '%%cellm line\nbody\n'
    out = self.sp.transform_cell(src)
    ref = "get_ipython().run_cell_magic('cellm', 'line', 'body')\n"
    assert out == ref