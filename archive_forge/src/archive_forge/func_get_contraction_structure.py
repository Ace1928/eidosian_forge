from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
def get_contraction_structure(expr):
    """Determine dummy indices of ``expr`` and describe its structure

    By *dummy* we mean indices that are summation indices.

    The structure of the expression is determined and described as follows:

    1) A conforming summation of Indexed objects is described with a dict where
       the keys are summation indices and the corresponding values are sets
       containing all terms for which the summation applies.  All Add objects
       in the SymPy expression tree are described like this.

    2) For all nodes in the SymPy expression tree that are *not* of type Add, the
       following applies:

       If a node discovers contractions in one of its arguments, the node
       itself will be stored as a key in the dict.  For that key, the
       corresponding value is a list of dicts, each of which is the result of a
       recursive call to get_contraction_structure().  The list contains only
       dicts for the non-trivial deeper contractions, omitting dicts with None
       as the one and only key.

    .. Note:: The presence of expressions among the dictionary keys indicates
       multiple levels of index contractions.  A nested dict displays nested
       contractions and may itself contain dicts from a deeper level.  In
       practical calculations the summation in the deepest nested level must be
       calculated first so that the outer expression can access the resulting
       indexed object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import get_contraction_structure
    >>> from sympy import default_sort_key
    >>> from sympy.tensor import IndexedBase, Idx
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, k, l = map(Idx, ['i', 'j', 'k', 'l'])
    >>> get_contraction_structure(x[i]*y[i] + A[j, j])
    {(i,): {x[i]*y[i]}, (j,): {A[j, j]}}
    >>> get_contraction_structure(x[i]*y[j])
    {None: {x[i]*y[j]}}

    A multiplication of contracted factors results in nested dicts representing
    the internal contractions.

    >>> d = get_contraction_structure(x[i, i]*y[j, j])
    >>> sorted(d.keys(), key=default_sort_key)
    [None, x[i, i]*y[j, j]]

    In this case, the product has no contractions:

    >>> d[None]
    {x[i, i]*y[j, j]}

    Factors are contracted "first":

    >>> sorted(d[x[i, i]*y[j, j]], key=default_sort_key)
    [{(i,): {x[i, i]}}, {(j,): {y[j, j]}}]

    A parenthesized Add object is also returned as a nested dictionary.  The
    term containing the parenthesis is a Mul with a contraction among the
    arguments, so it will be found as a key in the result.  It stores the
    dictionary resulting from a recursive call on the Add expression.

    >>> d = get_contraction_structure(x[i]*(y[i] + A[i, j]*x[j]))
    >>> sorted(d.keys(), key=default_sort_key)
    [(A[i, j]*x[j] + y[i])*x[i], (i,)]
    >>> d[(i,)]
    {(A[i, j]*x[j] + y[i])*x[i]}
    >>> d[x[i]*(A[i, j]*x[j] + y[i])]
    [{None: {y[i]}, (j,): {A[i, j]*x[j]}}]

    Powers with contractions in either base or exponent will also be found as
    keys in the dictionary, mapping to a list of results from recursive calls:

    >>> d = get_contraction_structure(A[j, j]**A[i, i])
    >>> d[None]
    {A[j, j]**A[i, i]}
    >>> nested_contractions = d[A[j, j]**A[i, i]]
    >>> nested_contractions[0]
    {(j,): {A[j, j]}}
    >>> nested_contractions[1]
    {(i,): {A[i, i]}}

    The description of the contraction structure may appear complicated when
    represented with a string in the above examples, but it is easy to iterate
    over:

    >>> from sympy import Expr
    >>> for key in d:
    ...     if isinstance(key, Expr):
    ...         continue
    ...     for term in d[key]:
    ...         if term in d:
    ...             # treat deepest contraction first
    ...             pass
    ...     # treat outermost contactions here

    """
    if isinstance(expr, Indexed):
        junk, key = _remove_repeated(expr.indices)
        return {key or None: {expr}}
    elif expr.is_Atom:
        return {None: {expr}}
    elif expr.is_Mul:
        junk, junk, key = _get_indices_Mul(expr, return_dummies=True)
        result = {key or None: {expr}}
        nested = []
        for fac in expr.args:
            facd = get_contraction_structure(fac)
            if not (None in facd and len(facd) == 1):
                nested.append(facd)
        if nested:
            result[expr] = nested
        return result
    elif expr.is_Pow or isinstance(expr, exp):
        b, e = expr.as_base_exp()
        dbase = get_contraction_structure(b)
        dexp = get_contraction_structure(e)
        dicts = []
        for d in (dbase, dexp):
            if not (None in d and len(d) == 1):
                dicts.append(d)
        result = {None: {expr}}
        if dicts:
            result[expr] = dicts
        return result
    elif expr.is_Add:
        result = {}
        for term in expr.args:
            d = get_contraction_structure(term)
            for key in d:
                if key in result:
                    result[key] |= d[key]
                else:
                    result[key] = d[key]
        return result
    elif isinstance(expr, Piecewise):
        return {None: expr}
    elif isinstance(expr, Function):
        deeplist = []
        for arg in expr.args:
            deep = get_contraction_structure(arg)
            if not (None in deep and len(deep) == 1):
                deeplist.append(deep)
        d = {None: {expr}}
        if deeplist:
            d[expr] = deeplist
        return d
    elif not expr.has(Indexed):
        return {None: {expr}}
    raise NotImplementedError('FIXME: No specialized handling of type %s' % type(expr))