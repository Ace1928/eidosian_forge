from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.singleton import S, Singleton
def test_singleton_redefinition():

    class TestSingleton(Basic, metaclass=Singleton):
        pass
    assert TestSingleton() is S.TestSingleton

    class TestSingleton(Basic, metaclass=Singleton):
        pass
    assert TestSingleton() is S.TestSingleton