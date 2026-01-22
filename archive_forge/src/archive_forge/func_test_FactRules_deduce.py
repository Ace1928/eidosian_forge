from sympy.core.facts import (deduce_alpha_implications,
from sympy.core.logic import And, Not
from sympy.testing.pytest import raises
def test_FactRules_deduce():
    f = FactRules(['a -> b', 'b -> c', 'b -> d', 'c -> e'])

    def D(facts):
        kb = FactKB(f)
        kb.deduce_all_facts(facts)
        return kb
    assert D({'a': T}) == {'a': T, 'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'b': T}) == {'b': T, 'c': T, 'd': T, 'e': T}
    assert D({'c': T}) == {'c': T, 'e': T}
    assert D({'d': T}) == {'d': T}
    assert D({'e': T}) == {'e': T}
    assert D({'a': F}) == {'a': F}
    assert D({'b': F}) == {'a': F, 'b': F}
    assert D({'c': F}) == {'a': F, 'b': F, 'c': F}
    assert D({'d': F}) == {'a': F, 'b': F, 'd': F}
    assert D({'a': U}) == {'a': U}