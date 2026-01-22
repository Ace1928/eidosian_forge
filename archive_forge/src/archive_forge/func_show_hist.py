from collections import OrderedDict
from itertools import zip_longest
import logging
import warnings
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import guard_transform
def show_hist(source, bins=10, masked=True, title='Histogram', ax=None, label=None, **kwargs):
    """Easily display a histogram with matplotlib.

    Parameters
    ----------
    source : array or dataset object opened in 'r' mode or Band or tuple(dataset, bidx)
        Input data to display.
        The first three arrays in multi-dimensional
        arrays are plotted as red, green, and blue.
    bins : int, optional
        Compute histogram across N bins.
    masked : bool, optional
        When working with a `rasterio.Band()` object, specifies if the data
        should be masked on read.
    title : str, optional
        Title for the figure.
    ax : matplotlib axes (opt)
        The raster will be added to this axes if passed.
    label : matplotlib labels (opt)
        If passed, matplotlib will use this label list.
        Otherwise, a default label list will be automatically created
    **kwargs : optional keyword arguments
        These will be passed to the matplotlib hist method. See full list at:
        http://matplotlib.org/api/axes_api.html?highlight=imshow#matplotlib.axes.Axes.hist
    """
    plt = get_plt()
    if isinstance(source, DatasetReader):
        arr = source.read(masked=masked)
    elif isinstance(source, (tuple, rasterio.Band)):
        arr = source[0].read(source[1], masked=masked)
    else:
        arr = source
    rng = (np.nanmin(arr), np.nanmax(arr))
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr.flatten(), 0).T
        colors = ['gold']
    else:
        arr = arr.reshape(arr.shape[0], -1).T
        colors = ['red', 'green', 'blue', 'violet', 'gold', 'saddlebrown']
    if arr.shape[-1] > len(colors):
        n = arr.shape[-1] - len(colors)
        colors.extend(np.ndarray.tolist(plt.get_cmap('Accent')(np.linspace(0, 1, n))))
    else:
        colors = colors[:arr.shape[-1]]
    if label:
        labels = label
    elif isinstance(source, (tuple, rasterio.Band)):
        labels = [str(source[1])]
    else:
        labels = (str(i + 1) for i in range(len(arr)))
    if ax:
        show = False
    else:
        show = True
        ax = plt.gca()
    fig = ax.get_figure()
    ax.hist(arr, bins=bins, color=colors, label=labels, range=rng, **kwargs)
    ax.legend(loc='upper right')
    ax.set_title(title, fontweight='bold')
    ax.grid(True)
    ax.set_xlabel('DN')
    ax.set_ylabel('Frequency')
    if show:
        plt.show()