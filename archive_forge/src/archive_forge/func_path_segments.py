from matplotlib.path import Path
import numpy as np
import shapely.geometry as sgeom
def path_segments(path, **kwargs):
    """
    Create an array of vertices and a corresponding array of codes from a
    :class:`matplotlib.path.Path`.

    Parameters
    ----------
    path
        A :class:`matplotlib.path.Path` instance.

    Other Parameters
    ----------------
    kwargs
        See :func:`matplotlib.path.iter_segments` for details of the keyword
        arguments.

    Returns
    -------
    vertices, codes
        A (vertices, codes) tuple, where vertices is a numpy array of
        coordinates, and codes is a numpy array of matplotlib path codes.
        See :class:`matplotlib.path.Path` for information on the types of
        codes and their meanings.

    """
    pth = path.cleaned(**kwargs)
    return (pth.vertices[:-1, :], pth.codes[:-1])