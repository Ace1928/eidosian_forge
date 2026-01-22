import numpy as np

    Transform and interpolate a vector field to a regular grid in the
    target projection.

    Parameters
    ----------
    src_crs
        The :class:`~cartopy.crs.CRS` that represents the coordinate
        system the vectors are defined in.
    target_proj
        The :class:`~cartopy.crs.Projection` that represents the
        projection the vectors are to be transformed to.
    regrid_shape
        The regular grid dimensions. If a single integer then the grid
        will have that number of points in the x and y directions. A
        2-tuple of integers specify the size of the regular grid in the
        x and y directions respectively.
    x, y
        The x and y coordinates, in the source CRS coordinates,
        where the vector components are located.
    u, v
        The grid eastward and grid northward components of the
        vector field respectively. Their shapes must match.

    Other Parameters
    ----------------
    scalars
        Zero or more scalar fields to regrid along with the vector
        components. Each scalar field must have the same shape as the
        vector components.
    target_extent
        The extent in the target CRS that the grid should occupy, in the
        form ``(x-lower, x-upper, y-lower, y-upper)``. Defaults to cover
        the full extent of the vector field.

    Returns
    -------
    x_grid, y_grid
        The x and y coordinates of the regular grid points as
        2-dimensional arrays.
    u_grid, v_grid
        The eastward and northward components of the vector field on
        the regular grid.
    scalars_grid
        The scalar fields on the regular grid. The number of returned
        scalar fields is the same as the number that were passed in.

    