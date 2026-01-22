import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def raw_in(prompt):
    try:
        return next(ilines)
    except StopIteration:
        return ''