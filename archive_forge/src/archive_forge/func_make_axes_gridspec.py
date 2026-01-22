import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
@_docstring.interpd
def make_axes_gridspec(parent, *, location=None, orientation=None, fraction=0.15, shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The axes is placed in the figure of the *parent* axes, by resizing and
    repositioning *parent*.

    This function is similar to `.make_axes` and mostly compatible with it.
    Primary differences are

    - `.make_axes_gridspec` requires the *parent* to have a subplotspec.
    - `.make_axes` positions the axes in figure coordinates;
      `.make_axes_gridspec` positions it using a subplotspec.
    - `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
      replaces the parent gridspec with a new one.

    Parameters
    ----------
    parent : `~matplotlib.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
    loc_settings = _normalize_location_orientation(location, orientation)
    kwargs['orientation'] = loc_settings['orientation']
    location = kwargs['ticklocation'] = loc_settings['location']
    aspect0 = aspect
    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    pad = kwargs.pop('pad', loc_settings['pad'])
    wh_space = 2 * pad / (1 - pad)
    if location in ('left', 'right'):
        height_ratios = [(1 - anchor[1]) * (1 - shrink), shrink, anchor[1] * (1 - shrink)]
        if location == 'left':
            gs = parent.get_subplotspec().subgridspec(1, 2, wspace=wh_space, width_ratios=[fraction, 1 - fraction - pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(3, 1, hspace=0, height_ratios=height_ratios)[1]
        else:
            gs = parent.get_subplotspec().subgridspec(1, 2, wspace=wh_space, width_ratios=[1 - fraction - pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(3, 1, hspace=0, height_ratios=height_ratios)[1]
    else:
        width_ratios = [anchor[0] * (1 - shrink), shrink, (1 - anchor[0]) * (1 - shrink)]
        if location == 'bottom':
            gs = parent.get_subplotspec().subgridspec(2, 1, hspace=wh_space, height_ratios=[1 - fraction - pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect
        else:
            gs = parent.get_subplotspec().subgridspec(2, 1, hspace=wh_space, height_ratios=[fraction, 1 - fraction - pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect
    parent.set_subplotspec(ss_main)
    if panchor is not False:
        parent.set_anchor(panchor)
    fig = parent.get_figure()
    cax = fig.add_subplot(ss_cb, label='<colorbar>')
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')
    cax._colorbar_info = dict(location=location, parents=[parent], shrink=shrink, anchor=anchor, panchor=panchor, fraction=fraction, aspect=aspect0, pad=pad)
    return (cax, kwargs)