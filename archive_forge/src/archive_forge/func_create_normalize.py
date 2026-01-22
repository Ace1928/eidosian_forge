from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def create_normalize(vmin, vmax, base=np.e, scale=None):
    """Create a colormap normalizer.

    Parameters
    ----------
    scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`,
    optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>

    Returns
    -------
    norm : `matplotlib.colors.Normalize`
    """
    if scale is None:
        scale = 'linear'
    if scale == 'linear':
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    elif scale == 'log':
        if vmin <= 0:
            raise ValueError("`vmin` must be positive for `cmap_scale='log'`. Got {}".format(vmin))
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmin)
    elif scale == 'symlog':
        norm = mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax, base=10)
    elif scale == 'sqrt':
        norm = mpl.colors.PowerNorm(gamma=1.0 / 2.0)
    elif isinstance(scale, mpl.colors.Normalize):
        norm = scale
    else:
        raise ValueError("Expected norm in ['linear', 'log', 'symlog','sqrt'] or a matplotlib.colors.Normalize object. Got {}".format(scale))
    return norm