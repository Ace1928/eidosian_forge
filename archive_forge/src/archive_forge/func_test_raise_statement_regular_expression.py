from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_raise_statement_regular_expression():
    candidates_ok = ["some text # raise Exception, 'text'", "raise ValueError('text') # raise Exception, 'text'", "raise ValueError('text')", 'raise ValueError', "raise ValueError('text')", "raise ValueError('text') #,", '\'"""This function will raise ValueError, except when it doesn\'t"""', "raise (ValueError('text')"]
    str_candidates_fail = ["raise 'exception'", "raise 'Exception'", 'raise "exception"', 'raise "Exception"', "raise 'ValueError'"]
    gen_candidates_fail = ["raise Exception('text') # raise Exception, 'text'", "raise Exception('text')", 'raise Exception', "raise Exception('text')", "raise Exception('text') #,", "raise Exception, 'text'", "raise Exception, 'text' # raise Exception('text')", "raise Exception, 'text' # raise Exception, 'text'", ">>> raise Exception, 'text'", ">>> raise Exception, 'text' # raise Exception('text')", ">>> raise Exception, 'text' # raise Exception, 'text'"]
    old_candidates_fail = ["raise Exception, 'text'", "raise Exception, 'text' # raise Exception('text')", "raise Exception, 'text' # raise Exception, 'text'", ">>> raise Exception, 'text'", ">>> raise Exception, 'text' # raise Exception('text')", ">>> raise Exception, 'text' # raise Exception, 'text'", "raise ValueError, 'text'", "raise ValueError, 'text' # raise Exception('text')", "raise ValueError, 'text' # raise Exception, 'text'", ">>> raise ValueError, 'text'", ">>> raise ValueError, 'text' # raise Exception('text')", ">>> raise ValueError, 'text' # raise Exception, 'text'", 'raise(ValueError,', 'raise (ValueError,', 'raise( ValueError,', 'raise ( ValueError,', 'raise(ValueError ,', 'raise (ValueError ,', 'raise( ValueError ,', 'raise ( ValueError ,']
    for c in candidates_ok:
        assert str_raise_re.search(_with_space(c)) is None, c
        assert gen_raise_re.search(_with_space(c)) is None, c
        assert old_raise_re.search(_with_space(c)) is None, c
    for c in str_candidates_fail:
        assert str_raise_re.search(_with_space(c)) is not None, c
    for c in gen_candidates_fail:
        assert gen_raise_re.search(_with_space(c)) is not None, c
    for c in old_candidates_fail:
        assert old_raise_re.search(_with_space(c)) is not None, c