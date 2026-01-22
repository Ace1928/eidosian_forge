from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket
from .errors import LaTeXParsingError
def convert_func(func):
    if func.func_normal():
        if func.L_PAREN():
            arg = convert_func_arg(func.func_arg())
        else:
            arg = convert_func_arg(func.func_arg_noparens())
        name = func.func_normal().start.text[1:]
        if name in ['arcsin', 'arccos', 'arctan', 'arccsc', 'arcsec', 'arccot']:
            name = 'a' + name[3:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        if name in ['arsinh', 'arcosh', 'artanh']:
            name = 'a' + name[2:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        if name == 'exp':
            expr = sympy.exp(arg, evaluate=False)
        if name in ('log', 'lg', 'ln'):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr())
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == 'lg':
                base = 10
            elif name in ('ln', 'log'):
                base = sympy.E
            expr = sympy.log(arg, base, evaluate=False)
        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())
        if name in ['sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'sinh', 'cosh', 'tanh']:
            if func_pow == -1:
                name = 'a' + name
                should_pow = False
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)
        return expr
    elif func.LETTER() or func.SYMBOL():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)
        if func.subexpr():
            if func.subexpr().expr():
                subscript = convert_expr(func.subexpr().expr())
            else:
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            fname += '_{' + subscriptName + '}'
        if func.SINGLE_QUOTES():
            fname += func.SINGLE_QUOTES().getText()
        input_args = func.args()
        output_args = []
        while input_args.args():
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))
        return sympy.Function(fname)(*output_args)
    elif func.FUNC_INT():
        return handle_integral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            return sympy.root(expr, r, evaluate=False)
        else:
            return sympy.sqrt(expr, evaluate=False)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        return sympy.conjugate(expr, evaluate=False)
    elif func.FUNC_SUM():
        return handle_sum_or_prod(func, 'summation')
    elif func.FUNC_PROD():
        return handle_sum_or_prod(func, 'product')
    elif func.FUNC_LIM():
        return handle_limit(func)