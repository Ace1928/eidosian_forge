from sympy.printing.codeprinter import CodePrinter
from sympy.core import symbols
from sympy.core.symbol import Dummy
from sympy.testing.pytest import raises
def test_print_Symbol():
    x, y = symbols('x, if')
    p = setup_test_printer()
    assert p._print(x) == 'x'
    assert p._print(y) == 'if'
    p.reserved_words.update(['if'])
    assert p._print(y) == 'if_'
    p = setup_test_printer(error_on_reserved=True)
    p.reserved_words.update(['if'])
    with raises(ValueError):
        p._print(y)
    p = setup_test_printer(reserved_word_suffix='_He_Man')
    p.reserved_words.update(['if'])
    assert p._print(y) == 'if_He_Man'