import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(graphs=None)
def from_pandas_adjacency(df, create_using=None):
    """Returns a graph from Pandas DataFrame.

    The Pandas DataFrame is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    df : Pandas DataFrame
      An adjacency matrix representation of a graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of df corresponds to an edge from i to j.

    If `df` has a single data type for each entry it will be converted to an
    appropriate Python data type.

    If you have node attributes stored in a separate dataframe `df_nodes`,
    you can load those attributes to the graph `G` using the following code:

    ```
    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})
    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())
    ```

    If `df` has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_pandas_adjacency

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> df = pd.DataFrame([[1, 1], [2, 1]])
    >>> df
       0  1
    0  1  1
    1  2  1
    >>> G = nx.from_pandas_adjacency(df)
    >>> G.name = "Graph from pandas adjacency matrix"
    >>> print(G)
    Graph named 'Graph from pandas adjacency matrix' with 2 nodes and 3 edges
    """
    try:
        df = df[df.index]
    except Exception as err:
        missing = list(set(df.index).difference(set(df.columns)))
        msg = f'{missing} not in columns'
        raise nx.NetworkXError('Columns must match Indices.', msg) from err
    A = df.values
    G = from_numpy_array(A, create_using=create_using)
    nx.relabel.relabel_nodes(G, dict(enumerate(df.columns)), copy=False)
    return G