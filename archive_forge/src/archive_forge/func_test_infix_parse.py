from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test_infix_parse():
    ops = [Operator('+', 2, 10), Operator('*', 2, 20), Operator('-', 1, 30)]
    atomic = ['ATOM1', 'ATOM2']
    mock_origin = Origin('asdf', 2, 3)
    tokens = [Token('ATOM1', mock_origin, 'a'), Token('+', mock_origin, '+'), Token('-', mock_origin, '-'), Token('ATOM2', mock_origin, 'b'), Token('*', mock_origin, '*'), Token(Token.LPAREN, mock_origin, '('), Token('ATOM1', mock_origin, 'c'), Token('+', mock_origin, '+'), Token('ATOM2', mock_origin, 'd'), Token(Token.RPAREN, mock_origin, ')')]
    tree = infix_parse(tokens, ops, atomic)

    def te(tree, type, extra):
        assert tree.type == type
        assert tree.token.extra == extra
    te(tree, '+', '+')
    te(tree.args[0], 'ATOM1', 'a')
    assert tree.args[0].args == []
    te(tree.args[1], '*', '*')
    te(tree.args[1].args[0], '-', '-')
    assert len(tree.args[1].args[0].args) == 1
    te(tree.args[1].args[0].args[0], 'ATOM2', 'b')
    te(tree.args[1].args[1], '+', '+')
    te(tree.args[1].args[1].args[0], 'ATOM1', 'c')
    te(tree.args[1].args[1].args[1], 'ATOM2', 'd')
    import pytest
    pytest.raises(ValueError, infix_parse, [], [Operator('+', 3, 10)], ['ATOMIC'])
    infix_parse(tokens, ops, atomic, trace=True)