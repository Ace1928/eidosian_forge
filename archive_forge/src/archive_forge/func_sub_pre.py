from sympy.core import Add, Basic, Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.traversal import preorder_traversal
def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
    adds = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]
    reps = {}
    ignore = set()
    for a in adds:
        na = -a
        if na.is_Mul:
            ignore.add(a)
            continue
        reps[a] = Mul._from_args([S.NegativeOne, na])
    e = e.xreplace(reps)
    if isinstance(e, Basic):
        negs = {}
        for a in sorted(e.atoms(Add), key=default_sort_key):
            if a in ignore:
                continue
            if a in reps:
                negs[a] = reps[a]
            elif a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        e = e.xreplace(negs)
    return e