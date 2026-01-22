import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
@pytest.mark.xfail(reason='Bug in python 3.9.8 –\xa0bpo 45738', condition=sys.version_info in [(3, 11, 0, 'alpha', 2)], raises=SystemError, strict=True)
def test_line_continuation(self):
    """ Test issue #2108."""
    isp = self.isp
    isp.push('1 \\\n\n')
    self.assertEqual(isp.push_accepts_more(), False)
    isp.push('1 \\ ')
    self.assertEqual(isp.push_accepts_more(), False)
    isp.push('(1 \\ ')
    self.assertEqual(isp.push_accepts_more(), False)