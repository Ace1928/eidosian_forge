from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def seifert_matrix(link, return_matrix_of_types=False):
    """
    Returns the Seifert matrix of a link by first making it isotopic to a braid
    closure.

    >>> K8n1 = [(8,6,9,5),(12,8,13,7),(1,4,2,5),(13,2,14,3),(3,14,4,15),
    ...         (15,10,0,11),(6,12,7,11),(9,0,10,1)]
    >>> L = Link(K8n1)
    >>> seifert_matrix(L)  # doctest: +NORMALIZE_WHITESPACE
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [-1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
     [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, -1, 1, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0]]

    Uses the algorithm described in:

    J. Collins, "An algorithm for computing the Seifert matrix of a link
    from a braid representation." (2007).
    """
    arrows = braid_arrows(link)
    strands = set((x[1] for x in arrows))
    grouped_by_strand = [[x for x in arrows if x[1] == strand] for strand in strands]
    hom_gens = [[(group[i][0], group[i + 1][0], group[i][2], group[i + 1][2]) for i in range(len(group) - 1)] for group in grouped_by_strand]
    num_gens = sum(map(len, hom_gens))
    matrix = [[0] * num_gens for i in range(num_gens)]
    entries = [(i, j) for i, hgi in enumerate(hom_gens) for j in range(len(hgi))]
    type_matrix = [[0] * num_gens for i in range(num_gens)]
    for n, strand in enumerate(hom_gens):
        for m, gen in enumerate(strand):
            if gen[2] == gen[3]:
                if gen[2] == 0:
                    matrix[entries.index((n, m))][entries.index((n, m))] = -1
                    type_matrix[entries.index((n, m))][entries.index((n, m))] = 1
                else:
                    matrix[entries.index((n, m))][entries.index((n, m))] = 1
                    type_matrix[entries.index((n, m))][entries.index((n, m))] = 2
        for m, gen in enumerate(strand[:-1]):
            if gen[3] == 0:
                matrix[entries.index((n, m + 1))][entries.index((n, m))] = 1
                type_matrix[entries.index((n, m + 1))][entries.index((n, m))] = 3
            else:
                matrix[entries.index((n, m))][entries.index((n, m + 1))] = -1
                type_matrix[entries.index((n, m))][entries.index((n, m + 1))] = 4
        if n != len(hom_gens) - 1:
            next_strand = hom_gens[n + 1]
            for m, gen in enumerate(strand):
                for l, next_gen in enumerate(next_strand):
                    if next_gen[0] < gen[0] < next_gen[1] < gen[1]:
                        matrix[entries.index((n + 1, l))][entries.index((n, m))] = 1
                        type_matrix[entries.index((n + 1, l))][entries.index((n, m))] = 5
                    elif gen[0] < next_gen[0] < gen[1] < next_gen[1]:
                        matrix[entries.index((n + 1, l))][entries.index((n, m))] = -1
                        type_matrix[entries.index((n + 1, l))][entries.index((n, m))] = 6
    if return_matrix_of_types:
        return (matrix, type_matrix)
    else:
        return matrix