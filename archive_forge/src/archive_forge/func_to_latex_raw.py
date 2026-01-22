import numbers
import os
import networkx as nx
@nx.utils.not_implemented_for('multigraph')
def to_latex_raw(G, pos='pos', tikz_options='', default_node_options='', node_options='node_options', node_label='label', default_edge_options='', edge_options='edge_options', edge_label='label', edge_label_options='edge_label_options'):
    """Return a string of the LaTeX/TikZ code to draw `G`

    This function produces just the code for the tikzpicture
    without any enclosing environment.

    Parameters
    ==========
    G : NetworkX graph
        The NetworkX graph to be drawn
    pos : string or dict (default "pos")
        The name of the node attribute on `G` that holds the position of each node.
        Positions can be sequences of length 2 with numbers for (x,y) coordinates.
        They can also be strings to denote positions in TikZ style, such as (x, y)
        or (angle:radius).
        If a dict, it should be keyed by node to a position.
        If an empty dict, a circular layout is computed by TikZ.
    tikz_options : string
        The tikzpicture options description defining the options for the picture.
        Often large scale options like `[scale=2]`.
    default_node_options : string
        The draw options for a path of nodes. Individual node options override these.
    node_options : string or dict
        The name of the node attribute on `G` that holds the options for each node.
        Or a dict keyed by node to a string holding the options for that node.
    node_label : string or dict
        The name of the node attribute on `G` that holds the node label (text)
        displayed for each node. If the attribute is "" or not present, the node
        itself is drawn as a string. LaTeX processing such as ``"$A_1$"`` is allowed.
        Or a dict keyed by node to a string holding the label for that node.
    default_edge_options : string
        The options for the scope drawing all edges. The default is "[-]" for
        undirected graphs and "[->]" for directed graphs.
    edge_options : string or dict
        The name of the edge attribute on `G` that holds the options for each edge.
        If the edge is a self-loop and ``"loop" not in edge_options`` the option
        "loop," is added to the options for the self-loop edge. Hence you can
        use "[loop above]" explicitly, but the default is "[loop]".
        Or a dict keyed by edge to a string holding the options for that edge.
    edge_label : string or dict
        The name of the edge attribute on `G` that holds the edge label (text)
        displayed for each edge. If the attribute is "" or not present, no edge
        label is drawn.
        Or a dict keyed by edge to a string holding the label for that edge.
    edge_label_options : string or dict
        The name of the edge attribute on `G` that holds the label options for
        each edge. For example, "[sloped,above,blue]". The default is no options.
        Or a dict keyed by edge to a string holding the label options for that edge.

    Returns
    =======
    latex_code : string
       The text string which draws the desired graph(s) when compiled by LaTeX.

    See Also
    ========
    to_latex
    write_latex
    """
    i4 = '\n    '
    i8 = '\n        '
    if not isinstance(pos, dict):
        pos = nx.get_node_attributes(G, pos)
    if not pos:
        pos = {n: f'({round(360.0 * i / len(G), 3)}:2)' for i, n in enumerate(G)}
    for node in G:
        if node not in pos:
            raise nx.NetworkXError(f'node {node} has no specified pos {pos}')
        posnode = pos[node]
        if not isinstance(posnode, str):
            try:
                posx, posy = posnode
                pos[node] = f'({round(posx, 3)}, {round(posy, 3)})'
            except (TypeError, ValueError):
                msg = f'position pos[{node}] is not 2-tuple or a string: {posnode}'
                raise nx.NetworkXError(msg)
    if not isinstance(node_options, dict):
        node_options = nx.get_node_attributes(G, node_options)
    if not isinstance(node_label, dict):
        node_label = nx.get_node_attributes(G, node_label)
    if not isinstance(edge_options, dict):
        edge_options = nx.get_edge_attributes(G, edge_options)
    if not isinstance(edge_label, dict):
        edge_label = nx.get_edge_attributes(G, edge_label)
    if not isinstance(edge_label_options, dict):
        edge_label_options = nx.get_edge_attributes(G, edge_label_options)
    topts = '' if tikz_options == '' else f'[{tikz_options.strip('[]')}]'
    defn = '' if default_node_options == '' else f'[{default_node_options.strip('[]')}]'
    linestyle = f'{('->' if G.is_directed() else '-')}'
    if default_edge_options == '':
        defe = '[' + linestyle + ']'
    elif '-' in default_edge_options:
        defe = default_edge_options
    else:
        defe = f'[{linestyle},{default_edge_options.strip('[]')}]'
    result = '  \\begin{tikzpicture}' + topts
    result += i4 + '  \\draw' + defn
    for n in G:
        nopts = f'[{node_options[n].strip('[]')}]' if n in node_options else ''
        ntext = f'{{{node_label[n]}}}' if n in node_label else f'{{{n}}}'
        result += i8 + f'{pos[n]} node{nopts} ({n}){ntext}'
    result += ';\n'
    result += '      \\begin{scope}' + defe
    for edge in G.edges:
        u, v = edge[:2]
        e_opts = f'{edge_options[edge]}'.strip('[]') if edge in edge_options else ''
        if u == v and 'loop' not in e_opts:
            e_opts = 'loop,' + e_opts
        e_opts = f'[{e_opts}]' if e_opts != '' else ''
        els = edge_label_options[edge] if edge in edge_label_options else ''
        els = f'[{els.strip('[]')}]'
        e_label = f' node{els} {{{edge_label[edge]}}}' if edge in edge_label else ''
        result += i8 + f'\\draw{e_opts} ({u}) to{e_label} ({v});'
    result += '\n      \\end{scope}\n    \\end{tikzpicture}\n'
    return result