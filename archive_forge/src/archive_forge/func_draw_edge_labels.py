from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number
import numpy as np
import rustworkx
def draw_edge_labels(graph, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, rotate=True, clip_on=True):
    """Draw edge labels.

    Parameters
    ----------
    graph: A rustworkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

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
        Vertical alignment {'center', 'top', 'bottom', 'baseline',
                            'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running rustworkx.visualization.mpl_draw(). You can install matplotlib with:\n'pip install matplotlib'") from e
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in graph.weighted_edge_list()}
    else:
        labels = edge_labels
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
            bbox = dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)
        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=trans_angle, transform=ax.transData, bbox=bbox, zorder=1, clip_on=clip_on)
        text_items[n1, n2] = t
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return text_items