from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def generate_colorbar(cmap=None, vmin=None, vmax=None, scale=None, ax=None, title=None, title_rotation=270, fontsize=None, n_ticks='auto', labelpad=10, mappable=None, **kwargs):
    """Generate a colorbar on an axis.

    Parameters
    ----------
    cmap : `matplotlib` colormap or str
        Colormap with which to draw colorbar
    vmin, vmax : float, optional (default: None)
        Range of values to display on colorbar
    scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`,
        optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    ax : `matplotlib.axes.Axes`, list or None, optional (default: None)
        Axis or list of axes from which to steal space for colorbar
        If `None`, uses the current axis
    title : str, optional (default: None)
        Title to display alongside colorbar
    title_rotation : int, optional (default: 270)
        Angle of rotation of the colorbar title
    fontsize : int, optional (default: None)
        Base font size.
    n_ticks : int, optional (default: 'auto')
        Maximum number of ticks. If the string 'auto', the number of ticks will
        be automatically determined based on the length of the colorbar.
    labelpad : scalar, optional, default: 10
        Spacing in points between the label and the x-axis.
    mappable : matplotlib.cm.ScalarMappable, optional (default: None)
        matplotlib mappable object (e.g. an axis image)
        from which to generate the colorbar

    kwargs : additional arguments for `plt.colorbar`

    Returns
    -------
    colorbar : `matplotlib.colorbar.Colorbar`
    """
    with temp_fontsize(fontsize):
        try:
            plot_axis = ax[0]
        except TypeError:
            plot_axis = ax
        fig, plot_axis, _ = _get_figure(plot_axis)
        if mappable is None:
            if vmax is None and vmin is None:
                vmax = 1
                vmin = 0
                color_range = np.linspace(vmin, vmax, 10).reshape(-1, 1)
                remove_ticks = True
                norm = None
                if n_ticks != 'auto':
                    warnings.warn('Cannot set `n_ticks` without setting `vmin` and `vmax`.', UserWarning)
            elif vmax is None or vmin is None:
                raise ValueError('Either both or neither of `vmax` and `vmin` should be set. Got `vmax={}, vmin={}`'.format(vmax, vmin))
            else:
                remove_ticks = False
                norm = create_normalize(vmin, vmax, scale=scale)
                color_range = np.linspace(vmin, vmax, 10).reshape(-1, 1)
                vmax = vmin = None
            if ax is None:
                ax = plot_axis
            xmin, xmax = plot_axis.get_xlim()
            ymin, ymax = plot_axis.get_ylim()
            if hasattr(cmap, '__len__') and (not isinstance(cmap, (str, dict))):
                cmap = create_colormap(cmap)
            mappable = plot_axis.imshow(color_range, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, aspect='auto', origin='lower', extent=[xmin, xmax, ymin, ymax])
            mappable.remove()
        else:
            if vmin is not None or vmax is not None:
                warnings.warn('Cannot set `vmin` or `vmax` when `mappable` is given.', UserWarning)
            if cmap is not None:
                warnings.warn('Cannot set `cmap` when `mappable` is given.', UserWarning)
            if scale is not None:
                warnings.warn('Cannot set `scale` when `mappable` is given.', UserWarning)
            remove_ticks = False
        colorbar = fig.colorbar(mappable, ax=ax, **kwargs)
        if remove_ticks or n_ticks == 0:
            colorbar.set_ticks([])
            labelpad += plt.rcParams['font.size']
        else:
            if n_ticks != 'auto':
                tick_locator = mpl.ticker.MaxNLocator(nbins=n_ticks - 1)
                colorbar.locator = tick_locator
                colorbar.update_ticks()
            colorbar.ax.tick_params(labelsize=parse_fontsize(None, 'large'))
        if title is not None:
            title_fontsize = parse_fontsize(None, 'x-large')
            colorbar.set_label(title, rotation=title_rotation, fontsize=title_fontsize, labelpad=labelpad)
    return colorbar