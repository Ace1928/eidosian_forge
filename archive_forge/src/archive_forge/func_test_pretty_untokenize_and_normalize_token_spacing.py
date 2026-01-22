import tokenize
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
def test_pretty_untokenize_and_normalize_token_spacing():
    assert normalize_token_spacing('1 + 1') == '1 + 1'
    assert normalize_token_spacing('1+1') == '1 + 1'
    assert normalize_token_spacing('1*(2+3**2)') == '1 * (2 + 3 ** 2)'
    assert normalize_token_spacing('a and b') == 'a and b'
    assert normalize_token_spacing('foo(a=bar.baz[1:])') == 'foo(a=bar.baz[1:])'
    assert normalize_token_spacing('{"hi":foo[:]}') == '{"hi": foo[:]}'
    assert normalize_token_spacing('\'a\' "b" \'c\'') == '\'a\' "b" \'c\''
    assert normalize_token_spacing('"""a""" is 1 or 2==3') == '"""a""" is 1 or 2 == 3'
    assert normalize_token_spacing('foo ( * args )') == 'foo(*args)'
    assert normalize_token_spacing('foo ( a * args )') == 'foo(a * args)'
    assert normalize_token_spacing('foo ( ** args )') == 'foo(**args)'
    assert normalize_token_spacing('foo ( a ** args )') == 'foo(a ** args)'
    assert normalize_token_spacing('foo (1, * args )') == 'foo(1, *args)'
    assert normalize_token_spacing('foo (1, a * args )') == 'foo(1, a * args)'
    assert normalize_token_spacing('foo (1, ** args )') == 'foo(1, **args)'
    assert normalize_token_spacing('foo (1, a ** args )') == 'foo(1, a ** args)'
    assert normalize_token_spacing('a=foo(b = 1)') == 'a = foo(b=1)'
    assert normalize_token_spacing('foo(+ 10, bar = - 1)') == 'foo(+10, bar=-1)'
    assert normalize_token_spacing('1 + +10 + -1 - 5') == '1 + +10 + -1 - 5'