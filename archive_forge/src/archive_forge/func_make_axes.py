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
def make_axes(parents, location=None, orientation=None, fraction=0.15, shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The axes is placed in the figure of the *parents* axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
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
    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    aspect0 = aspect
    if isinstance(parents, np.ndarray):
        parents = list(parents.flat)
    elif np.iterable(parents):
        parents = list(parents)
    else:
        parents = [parents]
    fig = parents[0].get_figure()
    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kwargs.pop('pad', pad0)
    if not all((fig is ax.get_figure() for ax in parents)):
        raise ValueError('Unable to create a colorbar axes as not all parents share the same figure.')
    parents_bbox = mtransforms.Bbox.union([ax.get_position(original=True).frozen() for ax in parents])
    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)
        aspect = 1.0 / aspect
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if panchor is not False:
            ax.set_anchor(panchor)
    cax = fig.add_axes(pbcb, label='<colorbar>')
    for a in parents:
        a._colorbars += [cax]
    cax._colorbar_info = dict(parents=parents, location=location, shrink=shrink, anchor=anchor, panchor=panchor, fraction=fraction, aspect=aspect0, pad=pad)
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')
    return (cax, kwargs)