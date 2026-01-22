import networkx as nx
@nx._dispatch(graphs=None)
def number_of_nonisomorphic_trees(order):
    """Returns the number of nonisomorphic trees

    Parameters
    ----------
    order : int
      order of the desired tree(s)

    Returns
    -------
    length : Number of nonisomorphic graphs for the given order

    References
    ----------

    """
    return sum((1 for _ in nonisomorphic_trees(order)))