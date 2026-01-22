import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push_accepts_more5(self):
    isp = self.isp
    isp.push('try:')
    isp.push('    a = 5')
    isp.push('except:')
    isp.push('    raise')
    self.assertEqual(isp.push_accepts_more(), True)