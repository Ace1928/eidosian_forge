from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def get_default_bbox_extra_artists(self):
    """
        Return a default list of artists that are used for the bounding box
        calculation.

        Artists are excluded either by not being visible or
        ``artist.set_in_layout(False)``.
        """
    artists = self.get_children()
    for axis in self._axis_map.values():
        artists.remove(axis)
    if not (self.axison and self._frameon):
        for spine in self.spines.values():
            artists.remove(spine)
    artists.remove(self.title)
    artists.remove(self._left_title)
    artists.remove(self._right_title)
    noclip = (_AxesBase, maxis.Axis, offsetbox.AnnotationBbox, offsetbox.OffsetBox)
    return [a for a in artists if a.get_visible() and a.get_in_layout() and (isinstance(a, noclip) or not a._fully_clipped_to_axes())]