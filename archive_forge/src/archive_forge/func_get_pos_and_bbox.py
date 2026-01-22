import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def get_pos_and_bbox(ax, renderer):
    """
    Get the position and the bbox for the axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.

    Returns
    -------
    pos : `~matplotlib.transforms.Bbox`
        Position in figure coordinates.
    bbox : `~matplotlib.transforms.Bbox`
        Tight bounding box in figure coordinates.
    """
    fig = ax.figure
    pos = ax.get_position(original=True)
    pos = pos.transformed(fig.transSubfigure - fig.transFigure)
    tightbbox = martist._get_tightbbox_for_layout_only(ax, renderer)
    if tightbbox is None:
        bbox = pos
    else:
        bbox = tightbbox.transformed(fig.transFigure.inverted())
    return (pos, bbox)