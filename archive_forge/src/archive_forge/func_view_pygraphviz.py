import os
import tempfile
import networkx as nx
@nx.utils.open_file(5, 'w+b')
def view_pygraphviz(G, edgelabel=None, prog='dot', args='', suffix='', path=None, show=True):
    """Views the graph G using the specified layout algorithm.

    Parameters
    ----------
    G : NetworkX graph
        The machine to draw.
    edgelabel : str, callable, None
        If a string, then it specifies the edge attribute to be displayed
        on the edge labels. If a callable, then it is called for each
        edge and it should return the string to be displayed on the edges.
        The function signature of `edgelabel` should be edgelabel(data),
        where `data` is the edge attribute dictionary.
    prog : string
        Name of Graphviz layout program.
    args : str
        Additional arguments to pass to the Graphviz layout program.
    suffix : str
        If `filename` is None, we save to a temporary file.  The value of
        `suffix` will appear at the tail end of the temporary filename.
    path : str, None
        The filename used to save the image.  If None, save to a temporary
        file.  File formats are the same as those from pygraphviz.agraph.draw.
    show : bool, default = True
        Whether to display the graph with :mod:`PIL.Image.show`,
        default is `True`. If `False`, the rendered graph is still available
        at `path`.

    Returns
    -------
    path : str
        The filename of the generated image.
    A : PyGraphviz graph
        The PyGraphviz graph instance used to generate the image.

    Notes
    -----
    If this function is called in succession too quickly, sometimes the
    image is not displayed. So you might consider time.sleep(.5) between
    calls if you experience problems.

    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.

    """
    if not len(G):
        raise nx.NetworkXException('An empty graph cannot be drawn.')
    attrs = ['edge', 'node', 'graph']
    for attr in attrs:
        if attr not in G.graph:
            G.graph[attr] = {}
    edge_attrs = {'fontsize': '10'}
    node_attrs = {'style': 'filled', 'fillcolor': '#0000FF40', 'height': '0.75', 'width': '0.75', 'shape': 'circle'}
    graph_attrs = {}

    def update_attrs(which, attrs):
        added = []
        for k, v in attrs.items():
            if k not in G.graph[which]:
                G.graph[which][k] = v
                added.append(k)

    def clean_attrs(which, added):
        for attr in added:
            del G.graph[which][attr]
        if not G.graph[which]:
            del G.graph[which]
    update_attrs('edge', edge_attrs)
    update_attrs('node', node_attrs)
    update_attrs('graph', graph_attrs)
    A = to_agraph(G)
    clean_attrs('edge', edge_attrs)
    clean_attrs('node', node_attrs)
    clean_attrs('graph', graph_attrs)
    if edgelabel is not None:
        if not callable(edgelabel):

            def func(data):
                return ''.join(['  ', str(data[edgelabel]), '  '])
        else:
            func = edgelabel
        if G.is_multigraph():
            for u, v, key, data in G.edges(keys=True, data=True):
                edge = A.get_edge(u, v, str(key))
                edge.attr['label'] = str(func(data))
        else:
            for u, v, data in G.edges(data=True):
                edge = A.get_edge(u, v)
                edge.attr['label'] = str(func(data))
    if path is None:
        ext = 'png'
        if suffix:
            suffix = f'_{suffix}.{ext}'
        else:
            suffix = f'.{ext}'
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    else:
        pass
    A.draw(path=path, format=None, prog=prog, args=args)
    path.close()
    if show:
        from PIL import Image
        Image.open(path.name).show()
    return (path.name, A)