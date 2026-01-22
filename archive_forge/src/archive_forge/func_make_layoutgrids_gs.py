import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """
    if gs in layoutgrids or gs.figure is None:
        return layoutgrids
    layoutgrids['hasgrids'] = True
    if not hasattr(gs, '_subplot_spec'):
        parent = layoutgrids[gs.figure]
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(parent=parent, parent_inner=True, name='gridspec', ncols=gs._ncols, nrows=gs._nrows, width_ratios=gs.get_width_ratios(), height_ratios=gs.get_height_ratios())
    else:
        subplot_spec = gs._subplot_spec
        parentgs = subplot_spec.get_gridspec()
        if parentgs not in layoutgrids:
            layoutgrids = make_layoutgrids_gs(layoutgrids, parentgs)
        subspeclb = layoutgrids[parentgs]
        rep = (gs, 'top')
        if rep not in layoutgrids:
            layoutgrids[rep] = mlayoutgrid.LayoutGrid(parent=subspeclb, name='top', nrows=1, ncols=1, parent_pos=(subplot_spec.rowspan, subplot_spec.colspan))
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(parent=layoutgrids[rep], name='gridspec', nrows=gs._nrows, ncols=gs._ncols, width_ratios=gs.get_width_ratios(), height_ratios=gs.get_height_ratios())
    return layoutgrids