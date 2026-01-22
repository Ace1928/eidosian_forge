from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_satisfiable_all_models():
    from sympy.abc import A, B
    assert next(satisfiable(False, all_models=True)) is False
    assert list(satisfiable(A >> ~A & A, all_models=True)) == [False]
    assert list(satisfiable(True, all_models=True)) == [{true: true}]
    models = [{A: True, B: False}, {A: False, B: True}]
    result = satisfiable(A ^ B, all_models=True)
    models.remove(next(result))
    models.remove(next(result))
    raises(StopIteration, lambda: next(result))
    assert not models
    assert list(satisfiable(Equivalent(A, B), all_models=True)) == [{A: False, B: False}, {A: True, B: True}]
    models = [{A: False, B: False}, {A: False, B: True}, {A: True, B: True}]
    for model in satisfiable(A >> B, all_models=True):
        models.remove(model)
    assert not models
    from sympy.utilities.iterables import numbered_symbols
    from sympy.logic.boolalg import Or
    sym = numbered_symbols()
    X = [next(sym) for i in range(100)]
    result = satisfiable(Or(*X), all_models=True)
    for i in range(10):
        assert next(result)