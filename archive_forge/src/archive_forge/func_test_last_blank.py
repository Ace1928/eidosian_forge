import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_last_blank():
    assert isp.last_blank('') is False
    assert isp.last_blank('abc') is False
    assert isp.last_blank('abc\n') is False
    assert isp.last_blank('abc\na') is False
    assert isp.last_blank('\n') is True
    assert isp.last_blank('\n ') is True
    assert isp.last_blank('abc\n ') is True
    assert isp.last_blank('abc\n\n') is True
    assert isp.last_blank('abc\nd\n\n') is True
    assert isp.last_blank('abc\nd\ne\n\n') is True
    assert isp.last_blank('abc \n \n \n\n') is True