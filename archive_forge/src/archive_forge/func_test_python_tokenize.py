import tokenize
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
def test_python_tokenize():
    code = 'a + (foo * -1)'
    tokens = list(python_tokenize(code))
    expected = [(tokenize.NAME, 'a', Origin(code, 0, 1)), (tokenize.OP, '+', Origin(code, 2, 3)), (tokenize.OP, '(', Origin(code, 4, 5)), (tokenize.NAME, 'foo', Origin(code, 5, 8)), (tokenize.OP, '*', Origin(code, 9, 10)), (tokenize.OP, '-', Origin(code, 11, 12)), (tokenize.NUMBER, '1', Origin(code, 12, 13)), (tokenize.OP, ')', Origin(code, 13, 14))]
    assert tokens == expected
    code2 = 'a + (b'
    tokens2 = list(python_tokenize(code2))
    expected2 = [(tokenize.NAME, 'a', Origin(code2, 0, 1)), (tokenize.OP, '+', Origin(code2, 2, 3)), (tokenize.OP, '(', Origin(code2, 4, 5)), (tokenize.NAME, 'b', Origin(code2, 5, 6))]
    assert tokens2 == expected2
    import pytest
    pytest.raises(PatsyError, list, python_tokenize('a b # c'))
    import pytest
    pytest.raises(PatsyError, list, python_tokenize('a b "c'))