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
def match_common_args(func_class, funcs, opt_subs):
    """
    Recognize and extract common subexpressions of function arguments within a
    set of function calls. For instance, for the following function calls::

        x + z + y
        sin(x + y)

    this will extract a common subexpression of `x + y`::

        w = x + y
        w + z
        sin(w)

    The function we work with is assumed to be associative and commutative.

    Parameters
    ==========

    func_class: class
        The function class (e.g. Add, Mul)
    funcs: list of functions
        A list of function calls.
    opt_subs: dict
        A dictionary of substitutions which this function may update.
    """
    funcs = sorted(funcs, key=lambda f: len(f.args))
    arg_tracker = FuncArgTracker(funcs)
    changed = OrderedSet()
    for i in range(len(funcs)):
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(arg_tracker.func_to_argset[i], min_func_i=i + 1)
        common_arg_candidates = OrderedSet(sorted(common_arg_candidates_counts.keys(), key=lambda k: (common_arg_candidates_counts[k], k)))
        while common_arg_candidates:
            j = common_arg_candidates.pop(last=False)
            com_args = arg_tracker.func_to_argset[i].intersection(arg_tracker.func_to_argset[j])
            if len(com_args) <= 1:
                continue
            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                com_func = Unevaluated(func_class, arg_tracker.get_args_in_value_order(com_args))
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                changed.add(i)
            else:
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])
            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            changed.add(j)
            for k in arg_tracker.get_subset_candidates(com_args, common_arg_candidates):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                changed.add(k)
        if i in changed:
            opt_subs[funcs[i]] = Unevaluated(func_class, arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))
        arg_tracker.stop_arg_tracking(i)