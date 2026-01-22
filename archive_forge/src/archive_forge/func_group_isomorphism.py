import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def group_isomorphism(G, H, isomorphism=True):
    """
    Compute an isomorphism between 2 given groups.

    Parameters
    ==========

    G : A finite ``FpGroup`` or a ``PermutationGroup``.
        First group.

    H : A finite ``FpGroup`` or a ``PermutationGroup``
        Second group.

    isomorphism : bool
        This is used to avoid the computation of homomorphism
        when the user only wants to check if there exists
        an isomorphism between the groups.

    Returns
    =======

    If isomorphism = False -- Returns a boolean.
    If isomorphism = True  -- Returns a boolean and an isomorphism between `G` and `H`.

    Examples
    ========

    >>> from sympy.combinatorics import free_group, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.homomorphisms import group_isomorphism
    >>> from sympy.combinatorics.named_groups import DihedralGroup, AlternatingGroup

    >>> D = DihedralGroup(8)
    >>> p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    >>> P = PermutationGroup(p)
    >>> group_isomorphism(D, P)
    (False, None)

    >>> F, a, b = free_group("a, b")
    >>> G = FpGroup(F, [a**3, b**3, (a*b)**2])
    >>> H = AlternatingGroup(4)
    >>> (check, T) = group_isomorphism(G, H)
    >>> check
    True
    >>> T(b*a*b**-1*a**-1*b**-1)
    (0 2 3)

    Notes
    =====

    Uses the approach suggested by Robert Tarjan to compute the isomorphism between two groups.
    First, the generators of ``G`` are mapped to the elements of ``H`` and
    we check if the mapping induces an isomorphism.

    """
    if not isinstance(G, (PermutationGroup, FpGroup)):
        raise TypeError('The group must be a PermutationGroup or an FpGroup')
    if not isinstance(H, (PermutationGroup, FpGroup)):
        raise TypeError('The group must be a PermutationGroup or an FpGroup')
    if isinstance(G, FpGroup) and isinstance(H, FpGroup):
        G = simplify_presentation(G)
        H = simplify_presentation(H)
        if G.generators == H.generators and G.relators.sort() == H.relators.sort():
            if not isomorphism:
                return True
            return (True, homomorphism(G, H, G.generators, H.generators))
    _H = H
    g_order = G.order()
    h_order = H.order()
    if g_order is S.Infinity:
        raise NotImplementedError('Isomorphism methods are not implemented for infinite groups.')
    if isinstance(H, FpGroup):
        if h_order is S.Infinity:
            raise NotImplementedError('Isomorphism methods are not implemented for infinite groups.')
        _H, h_isomorphism = H._to_perm_group()
    if g_order != h_order or G.is_abelian != H.is_abelian:
        if not isomorphism:
            return False
        return (False, None)
    if not isomorphism:
        n = g_order
        if igcd(n, totient(n)) == 1:
            return True
    gens = list(G.generators)
    for subset in itertools.permutations(_H, len(gens)):
        images = list(subset)
        images.extend([_H.identity] * (len(G.generators) - len(images)))
        _images = dict(zip(gens, images))
        if _check_homomorphism(G, _H, _images):
            if isinstance(H, FpGroup):
                images = h_isomorphism.invert(images)
            T = homomorphism(G, H, G.generators, images, check=False)
            if T.is_isomorphism():
                if not isomorphism:
                    return True
                return (True, T)
    if not isomorphism:
        return False
    return (False, None)