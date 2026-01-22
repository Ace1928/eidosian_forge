from collections import Counter, defaultdict
import networkx as nx
@nx._dispatch(node_attrs='[node_attributes]', edge_attrs='[edge_attributes]')
def snap_aggregation(G, node_attributes, edge_attributes=(), prefix='Supernode-', supernode_attribute='group', superedge_attribute='types'):
    """Creates a summary graph based on attributes and connectivity.

    This function uses the Summarization by Grouping Nodes on Attributes
    and Pairwise edges (SNAP) algorithm for summarizing a given
    graph by grouping nodes by node attributes and their edge attributes
    into supernodes in a summary graph.  This name SNAP should not be
    confused with the Stanford Network Analysis Project (SNAP).

    Here is a high-level view of how this algorithm works:

    1) Group nodes by node attribute values.

    2) Iteratively split groups until all nodes in each group have edges
    to nodes in the same groups. That is, until all the groups are homogeneous
    in their member nodes' edges to other groups.  For example,
    if all the nodes in group A only have edge to nodes in group B, then the
    group is homogeneous and does not need to be split. If all nodes in group B
    have edges with nodes in groups {A, C}, but some also have edges with other
    nodes in B, then group B is not homogeneous and needs to be split into
    groups have edges with {A, C} and a group of nodes having
    edges with {A, B, C}.  This way, viewers of the summary graph can
    assume that all nodes in the group have the exact same node attributes and
    the exact same edges.

    3) Build the output summary graph, where the groups are represented by
    super-nodes. Edges represent the edges shared between all the nodes in each
    respective groups.

    A SNAP summary graph can be used to visualize graphs that are too large to display
    or visually analyze, or to efficiently identify sets of similar nodes with similar connectivity
    patterns to other sets of similar nodes based on specified node and/or edge attributes in a graph.

    Parameters
    ----------
    G: graph
        Networkx Graph to be summarized
    node_attributes: iterable, required
        An iterable of the node attributes used to group nodes in the summarization process. Nodes
        with the same values for these attributes will be grouped together in the summary graph.
    edge_attributes: iterable, optional
        An iterable of the edge attributes considered in the summarization process.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type.
    prefix: str
        The prefix used to denote supernodes in the summary graph. Defaults to 'Supernode-'.
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes. Defaults to 'group'.
    superedge_attribute: str
        The edge attribute for recording the edge types of multiple edges. Defaults to 'types'.

    Returns
    -------
    networkx.Graph: summary graph

    Examples
    --------
    SNAP aggregation takes a graph and summarizes it in the context of user-provided
    node and edge attributes such that a viewer can more easily extract and
    analyze the information represented by the graph

    >>> nodes = {
    ...     "A": dict(color="Red"),
    ...     "B": dict(color="Red"),
    ...     "C": dict(color="Red"),
    ...     "D": dict(color="Red"),
    ...     "E": dict(color="Blue"),
    ...     "F": dict(color="Blue"),
    ... }
    >>> edges = [
    ...     ("A", "E", "Strong"),
    ...     ("B", "F", "Strong"),
    ...     ("C", "E", "Weak"),
    ...     ("D", "F", "Weak"),
    ... ]
    >>> G = nx.Graph()
    >>> for node in nodes:
    ...     attributes = nodes[node]
    ...     G.add_node(node, **attributes)
    ...
    >>> for source, target, type in edges:
    ...     G.add_edge(source, target, type=type)
    ...
    >>> node_attributes = ('color', )
    >>> edge_attributes = ('type', )
    >>> summary_graph = nx.snap_aggregation(G, node_attributes=node_attributes, edge_attributes=edge_attributes)

    Notes
    -----
    The summary graph produced is called a maximum Attribute-edge
    compatible (AR-compatible) grouping.  According to [1]_, an
    AR-compatible grouping means that all nodes in each group have the same
    exact node attribute values and the same exact edges and
    edge types to one or more nodes in the same groups.  The maximal
    AR-compatible grouping is the grouping with the minimal cardinality.

    The AR-compatible grouping is the most detailed grouping provided by
    any of the SNAP algorithms.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    """
    edge_types = {edge: tuple((attrs.get(attr) for attr in edge_attributes)) for edge, attrs in G.edges.items()}
    if not G.is_directed():
        if G.is_multigraph():
            edges = [((v, u, k), etype) for (u, v, k), etype in edge_types.items()]
        else:
            edges = [((v, u), etype) for (u, v), etype in edge_types.items()]
        edge_types.update(edges)
    group_lookup = {node: tuple((attrs[attr] for attr in node_attributes)) for node, attrs in G.nodes.items()}
    groups = defaultdict(set)
    for node, node_type in group_lookup.items():
        groups[node_type].add(node)
    eligible_group_id, neighbor_info = _snap_eligible_group(G, groups, group_lookup, edge_types)
    while eligible_group_id:
        groups = _snap_split(groups, neighbor_info, group_lookup, eligible_group_id)
        eligible_group_id, neighbor_info = _snap_eligible_group(G, groups, group_lookup, edge_types)
    return _snap_build_graph(G, groups, node_attributes, edge_attributes, neighbor_info, edge_types, prefix, supernode_attribute, superedge_attribute)