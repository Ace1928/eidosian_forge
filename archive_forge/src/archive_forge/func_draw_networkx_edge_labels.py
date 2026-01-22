from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, rotate=True, clip_on=True):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default=None)
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (default=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
        try:
            u, v = next(iter(labels))
        except ValueError as err:
            raise nx.NetworkXError('draw_networkx_edge_labels does not support multiedges.') from err
        except StopIteration:
            pass
    text_items = {}
    for (n1, n2), label in labels.items():
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        x, y = (x1 * label_pos + x2 * (1.0 - label_pos), y1 * label_pos + y2 * (1.0 - label_pos))
        if rotate:
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)), xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        if bbox is None:
            bbox = {'boxstyle': 'round', 'ec': (1.0, 1.0, 1.0), 'fc': (1.0, 1.0, 1.0)}
        if not isinstance(label, str):
            label = str(label)
        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=trans_angle, transform=ax.transData, bbox=bbox, zorder=1, clip_on=clip_on)
        text_items[n1, n2] = t
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return text_items