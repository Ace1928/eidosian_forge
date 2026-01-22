from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
def set_viewlim_mode(self, mode):
    _api.check_in_list([None, 'equal', 'transform'], mode=mode)
    self._viewlim_mode = mode