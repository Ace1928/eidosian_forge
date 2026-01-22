import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_indent3(self):
    isp = self.isp
    isp.push('if 1:')
    isp.push('    x = (1+\n    2)')
    self.assertEqual(isp.get_indent_spaces(), 4)