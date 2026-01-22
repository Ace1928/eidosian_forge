from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
    """
        Set the aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            Possible values:

            =========   ==================================================
            value       description
            =========   ==================================================
            'auto'      automatic; fill the position rectangle with data.
            'equal'     adapt all the axes to have equal aspect ratios.
            'equalxy'   adapt the x and y axes to have equal aspect ratios.
            'equalxz'   adapt the x and z axes to have equal aspect ratios.
            'equalyz'   adapt the y and z axes to have equal aspect ratios.
            =========   ==================================================

        adjustable : None or {'box', 'datalim'}, optional
            If not *None*, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or 2-tuple of float, optional
            If not *None*, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way to
            specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect
        """
    _api.check_in_list(('auto', 'equal', 'equalxy', 'equalyz', 'equalxz'), aspect=aspect)
    super().set_aspect(aspect='auto', adjustable=adjustable, anchor=anchor, share=share)
    self._aspect = aspect
    if aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
        ax_indices = self._equal_aspect_axis_indices(aspect)
        view_intervals = np.array([self.xaxis.get_view_interval(), self.yaxis.get_view_interval(), self.zaxis.get_view_interval()])
        ptp = np.ptp(view_intervals, axis=1)
        if self._adjustable == 'datalim':
            mean = np.mean(view_intervals, axis=1)
            scale = max(ptp[ax_indices] / self._box_aspect[ax_indices])
            deltas = scale * self._box_aspect
            for i, set_lim in enumerate((self.set_xlim3d, self.set_ylim3d, self.set_zlim3d)):
                if i in ax_indices:
                    set_lim(mean[i] - deltas[i] / 2.0, mean[i] + deltas[i] / 2.0)
        else:
            box_aspect = np.array(self._box_aspect)
            box_aspect[ax_indices] = ptp[ax_indices]
            remaining_ax_indices = {0, 1, 2}.difference(ax_indices)
            if remaining_ax_indices:
                remaining = remaining_ax_indices.pop()
                old_diag = np.linalg.norm(self._box_aspect[ax_indices])
                new_diag = np.linalg.norm(box_aspect[ax_indices])
                box_aspect[remaining] *= new_diag / old_diag
            self.set_box_aspect(box_aspect)