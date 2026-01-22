import numpy as np
import cartopy.crs as ccrs

    Regrid the data array from the source projection to the target projection.

    Parameters
    ----------
    array
        The :class:`numpy.ndarray` of data to be regridded to the
        target projection.
    source_x_coords
        A 2-dimensional source projection :class:`numpy.ndarray` of
        x-direction sample points.
    source_y_coords
        A 2-dimensional source projection :class:`numpy.ndarray` of
        y-direction sample points.
    source_proj
        The source :class:`~cartopy.crs.Projection` instance.
    target_proj
        The target :class:`~cartopy.crs.Projection` instance.
    target_x_points
        A 2-dimensional target projection :class:`numpy.ndarray` of
        x-direction sample points.
    target_y_points
        A 2-dimensional target projection :class:`numpy.ndarray` of
        y-direction sample points.
    mask_extrapolated: optional
        Assume that the source coordinate is rectilinear and so mask the
        resulting target grid values which lie outside the source grid domain.
        Defaults to False.

    Returns
    -------
    new_array
        The data array regridded in the target projection.

    