import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(preserve_edge_attrs=True)
def to_pandas_edgelist(G, source='source', target='target', nodelist=None, dtype=None, edge_key=None):
    """Returns the graph edge list as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    source : str or int, optional
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int, optional
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    nodelist : list, optional
       Use only nodes specified in nodelist

    dtype : dtype, default None
        Use to create the DataFrame. Data type to force.
        Only a single dtype is allowed. If None, infer.

    edge_key : str or int or None, optional (default=None)
        A valid column name (string or integer) for the edge keys (for the
        multigraph case). If None, edge keys are not stored in the DataFrame.

    Returns
    -------
    df : Pandas DataFrame
       Graph edge list

    Examples
    --------
    >>> G = nx.Graph(
    ...     [
    ...         ("A", "B", {"cost": 1, "weight": 7}),
    ...         ("C", "E", {"cost": 9, "weight": 10}),
    ...     ]
    ... )
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"])
    >>> df[["source", "target", "cost", "weight"]]
      source target  cost  weight
    0      A      B     1       7
    1      C      E     9      10

    >>> G = nx.MultiGraph([('A', 'B', {'cost': 1}), ('A', 'B', {'cost': 9})])
    >>> df = nx.to_pandas_edgelist(G, nodelist=['A', 'C'], edge_key='ekey')
    >>> df[['source', 'target', 'cost', 'ekey']]
      source target  cost  ekey
    0      A      B     1     0
    1      A      B     9     1

    """
    import pandas as pd
    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)
    source_nodes = [s for s, _, _ in edgelist]
    target_nodes = [t for _, t, _ in edgelist]
    all_attrs = set().union(*(d.keys() for _, _, d in edgelist))
    if source in all_attrs:
        raise nx.NetworkXError(f'Source name {source!r} is an edge attr name')
    if target in all_attrs:
        raise nx.NetworkXError(f'Target name {target!r} is an edge attr name')
    nan = float('nan')
    edge_attr = {k: [d.get(k, nan) for _, _, d in edgelist] for k in all_attrs}
    if G.is_multigraph() and edge_key is not None:
        if edge_key in all_attrs:
            raise nx.NetworkXError(f'Edge key name {edge_key!r} is an edge attr name')
        edge_keys = [k for _, _, k in G.edges(keys=True)]
        edgelistdict = {source: source_nodes, target: target_nodes, edge_key: edge_keys}
    else:
        edgelistdict = {source: source_nodes, target: target_nodes}
    edgelistdict.update(edge_attr)
    return pd.DataFrame(edgelistdict, dtype=dtype)