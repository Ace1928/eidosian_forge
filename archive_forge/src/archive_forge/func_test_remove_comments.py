import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_remove_comments():
    tests = [('text', 'text'), ('text # comment', 'text '), ('text # comment\n', 'text \n'), ('text # comment \n', 'text \n'), ('line # c \nline\n', 'line \nline\n'), ('line # c \nline#c2  \nline\nline #c\n\n', 'line \nline\nline\nline \n\n')]
    tt.check_pairs(isp.remove_comments, tests)