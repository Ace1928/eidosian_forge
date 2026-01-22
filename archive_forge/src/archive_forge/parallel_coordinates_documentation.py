import holoviews as hv
import colorcet as cc
from ..backend_transforms import _transfer_opts_cur_backend
from ..util import with_hv_extension

    Parallel coordinates plotting.

    To show a set of points in an n-dimensional space, a backdrop is drawn
    consisting of n parallel lines. A point in n-dimensional space is
    represented as a polyline with vertices on the parallel axes; the
    position of the vertex on the i-th axis corresponds to the i-th coordinate
    of the point.

    Parameters
    ----------
    frame: DataFrame
    class_column: str
        Column name containing class names
    cols: list, optional
        A list of column names to use
    alpha: float, optional
        The transparency of the lines
    cmap/colormap: str or colormap object
        Colormap to use for groups

    Returns
    -------
    obj : HoloViews object
        The HoloViews representation of the plot.

    See Also
    --------
    pandas.plotting.parallel_coordinates : matplotlib version of this routine
    