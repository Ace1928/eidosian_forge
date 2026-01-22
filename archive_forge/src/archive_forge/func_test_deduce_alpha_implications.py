from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_deduce_alpha_implications():

    def D(i):
        I = deduce_alpha_implications(i)
        P = rules_2prereq({(k, True): {(v, True) for v in S} for k, S in I.items()})
        return (I, P)
    I, P = D([('a', 'b'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'): {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    I, P = D([('a', 'b'), ('b', 'c'), ('b', 'c')])
    assert I == {'a': {'b', 'c'}, 'b': {'c'}, Not('b'): {Not('a')}, Not('c'): {Not('a'), Not('b')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    assert D([('a', 'a'), ('a', 'a')]) == ({}, {})
    assert D([('a', 'b'), ('b', 'a')]) == ({'a': {'b'}, 'b': {'a'}, Not('a'): {Not('b')}, Not('b'): {Not('a')}}, {'a': {'b'}, 'b': {'a'}})
    raises(ValueError, lambda: D([('a', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', Not('a'))]))
    raises(ValueError, lambda: D([('a', 'b'), ('b', 'c'), ('b', 'na'), ('na', Not('a'))]))
    I, P = D([('a', Not('b')), ('c', 'b')])
    assert I == {'a': {Not('b'), Not('c')}, 'b': {Not('a')}, 'c': {'b', Not('a')}, Not('b'): {Not('c')}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    I, P = D([(Not('a'), 'b'), ('a', 'c')])
    assert I == {'a': {'c'}, Not('a'): {'b'}, Not('b'): {'a', 'c'}, Not('c'): {Not('a'), 'b'}}
    assert P == {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}
    I, P = D([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
    assert I == {'a': {'b', 'c', 'd', 'e'}, 'b': {'c', 'd', 'e'}, 'c': {'d', 'e'}, 'd': {'e'}, Not('b'): {Not('a')}, Not('c'): {Not('a'), Not('b')}, Not('d'): {Not('a'), Not('b'), Not('c')}, Not('e'): {Not('a'), Not('b'), Not('c'), Not('d')}}
    assert P == {'a': {'b', 'c', 'd', 'e'}, 'b': {'a', 'c', 'd', 'e'}, 'c': {'a', 'b', 'd', 'e'}, 'd': {'a', 'b', 'c', 'e'}, 'e': {'a', 'b', 'c', 'd'}}
    I, P = D([('rat', 'real'), ('int', 'rat')])
    assert I == {'int': {'rat', 'real'}, 'rat': {'real'}, Not('real'): {Not('rat'), Not('int')}, Not('rat'): {Not('int')}}
    assert P == {'rat': {'int', 'real'}, 'real': {'int', 'rat'}, 'int': {'rat', 'real'}}