from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    """
    if opt_subs is None:
        opt_subs = {}
    to_eliminate = set()
    seen_subexp = set()
    excluded_symbols = set()

    def _find_repeated(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return
        if isinstance(expr, RootOf):
            return
        if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order or isinstance(expr, (MatrixSymbol, MatrixElement))):
            if expr.is_Symbol:
                excluded_symbols.add(expr)
            return
        if iterable(expr):
            args = expr
        else:
            if expr in seen_subexp:
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    to_eliminate.add(expr)
                    return
            seen_subexp.add(expr)
            if expr in opt_subs:
                expr = opt_subs[expr]
            args = expr.args
        list(map(_find_repeated, args))
    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)
    symbols = (symbol for symbol in symbols if symbol not in excluded_symbols)
    replacements = []
    subs = {}

    def _rebuild(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr
        if not expr.args:
            return expr
        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr.args]
            return expr.func(*new_args)
        if expr in subs:
            return subs[expr]
        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args
        new_args = list(map(_rebuild, args))
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr
        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError('Symbols iterator ran out of symbols.')
            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows, orig_expr.cols)
            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym
        else:
            return new_expr
    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)
    return (replacements, reduced_exprs)