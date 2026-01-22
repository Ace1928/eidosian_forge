import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(graphs=None)
def is_valid_directed_joint_degree(in_degrees, out_degrees, nkk):
    """Checks whether the given directed joint degree input is realizable

    Parameters
    ----------
    in_degrees :  list of integers
        in degree sequence contains the in degrees of nodes.
    out_degrees : list of integers
        out degree sequence contains the out degrees of nodes.
    nkk  :  dictionary of dictionary of integers
        directed joint degree dictionary. for nodes of out degree k (first
        level of dict) and nodes of in degree l (second level of dict)
        describes the number of edges.

    Returns
    -------
    boolean
        returns true if given input is realizable, else returns false.

    Notes
    -----
    Here is the list of conditions that the inputs (in/out degree sequences,
    nkk) need to satisfy for simple directed graph realizability:

    - Condition 0: in_degrees and out_degrees have the same length
    - Condition 1: nkk[k][l]  is integer for all k,l
    - Condition 2: sum(nkk[k])/k = number of nodes with partition id k, is an
                   integer and matching degree sequence
    - Condition 3: number of edges and non-chords between k and l cannot exceed
                   maximum possible number of edges


    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    V = {}
    forbidden = {}
    if len(in_degrees) != len(out_degrees):
        return False
    for idx in range(len(in_degrees)):
        i = in_degrees[idx]
        o = out_degrees[idx]
        V[i, 0] = V.get((i, 0), 0) + 1
        V[o, 1] = V.get((o, 1), 0) + 1
        forbidden[o, i] = forbidden.get((o, i), 0) + 1
    S = {}
    for k in nkk:
        for l in nkk[k]:
            val = nkk[k][l]
            if not float(val).is_integer():
                return False
            if val > 0:
                S[k, 1] = S.get((k, 1), 0) + val
                S[l, 0] = S.get((l, 0), 0) + val
                if val + forbidden.get((k, l), 0) > V[k, 1] * V[l, 0]:
                    return False
    return all((S[s] / s[0] == V[s] for s in S))