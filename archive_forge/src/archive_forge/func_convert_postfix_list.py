from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise LaTeXParsingError('Index out of bounds')
    res = convert_postfix(arr[i])
    if isinstance(res, sympy.Expr):
        if i == len(arr) - 1:
            return res
        else:
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, sympy.Expr) and isinstance(right, sympy.Expr):
                    left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                    right_syms = convert_postfix(arr[i + 1]).atoms(sympy.Symbol)
                    if not (left_syms or right_syms) and str(res) == 'x':
                        return convert_postfix_list(arr, i + 1)
            return sympy.Mul(res, convert_postfix_list(arr, i + 1), evaluate=False)
    else:
        wrt = res[0]
        if i == len(arr) - 1:
            raise LaTeXParsingError('Expected expression for derivative')
        else:
            expr = convert_postfix_list(arr, i + 1)
            return sympy.Derivative(expr, wrt)