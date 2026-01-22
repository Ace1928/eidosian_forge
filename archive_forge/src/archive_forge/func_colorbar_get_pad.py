import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def colorbar_get_pad(layoutgrids, cax):
    parents = cax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    cb_rspans, cb_cspans = get_cb_parent_spans(cax)
    bboxouter = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)
    if cax._colorbar_info['location'] in ['right', 'left']:
        size = bboxouter.width
    else:
        size = bboxouter.height
    return cax._colorbar_info['pad'] * size