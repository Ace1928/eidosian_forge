import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_indent4(self):
    isp = self.isp
    isp.push('if 1: \n    x=1')
    self.assertEqual(isp.get_indent_spaces(), 4)
    isp.push('y=2\n')
    self.assertEqual(isp.get_indent_spaces(), 0)
    isp.push('if 1:\t\n    x=1')
    self.assertEqual(isp.get_indent_spaces(), 4)
    isp.push('y=2\n')
    self.assertEqual(isp.get_indent_spaces(), 0)