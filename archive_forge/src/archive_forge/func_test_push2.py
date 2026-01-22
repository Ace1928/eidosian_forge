import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push2(self):
    isp = self.isp
    self.assertEqual(isp.push('if 1:'), False)
    for line in ['  x=1', '# a comment', '  y=2']:
        print(line)
        self.assertEqual(isp.push(line), True)