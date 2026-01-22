from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def tensor_gens(base, gens, list_free_indices, sym=0):
    """
    Returns size, res_base, res_gens BSGS for n tensors of the
    same type.

    Explanation
    ===========

    base, gens BSGS for tensors of this type
    list_free_indices  list of the slots occupied by fixed indices
                       for each of the tensors

    sym symmetry under commutation of two tensors
    sym   None  no symmetry
    sym   0     commuting
    sym   1     anticommuting

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs

    two symmetric tensors with 3 indices without free indices

    >>> base, gens = get_symmetric_group_sgs(3)
    >>> tensor_gens(base, gens, [[], []])
    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])

    two symmetric tensors with 3 indices with free indices in slot 1 and 0

    >>> tensor_gens(base, gens, [[1], [0]])
    (8, [0, 4], [(7)(0 2), (7)(4 5)])

    four symmetric tensors with 3 indices, two of which with free indices

    """

    def _get_bsgs(G, base, gens, free_indices):
        """
        return the BSGS for G.pointwise_stabilizer(free_indices)
        """
        if not free_indices:
            return (base[:], gens[:])
        else:
            H = G.pointwise_stabilizer(free_indices)
            base, sgs = H.schreier_sims_incremental()
            return (base, sgs)
    if not base and list_free_indices.count([]) < 2:
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        return (size, [], [_af_new(list(range(size)))])
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    res_base, res_gens = _get_bsgs(G, base, gens, list_free_indices[0])
    for i in range(1, len(list_free_indices)):
        base1, gens1 = _get_bsgs(G, base, gens, list_free_indices[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base1, gens1, 1)
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    res_gens = [h for h in res_gens if h._array_form != id_af]
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return (size, res_base, res_gens)
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    for i in base_comm:
        if i not in res_base:
            res_base.append(i)
    res_base.sort()
    if not res_gens:
        res_gens = [_af_new(id_af)]
    return (size, res_base, res_gens)