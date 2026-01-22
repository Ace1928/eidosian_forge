import warnings
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, color_from_dim, set_xticklabels, vectorized_to_hex
from . import backend_show, create_axes_grid, matplotlib_kwarg_dealiaser
def update_annot(ind):
    idx = ind['ind'][0]
    pos = sc_plot.get_offsets()[idx]
    annot_text = hover_format.format(idx, coord_labels[idx])
    annot.xy = pos
    annot.set_position((-offset if pos[0] > xmid else offset, -offset if pos[1] > ymid else offset))
    annot.set_text(annot_text)
    annot.get_bbox_patch().set_facecolor(rgba_c[idx])
    annot.set_ha('right' if pos[0] > xmid else 'left')
    annot.set_va('top' if pos[1] > ymid else 'bottom')