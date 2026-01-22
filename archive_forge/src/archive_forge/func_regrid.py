import numpy as np
import cartopy.crs as ccrs
def regrid(array, source_x_coords, source_y_coords, source_proj, target_proj, target_x_points, target_y_points, mask_extrapolated=False):
    """
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

    """
    xyz = source_proj.transform_points(source_proj, source_x_coords.flatten(), source_y_coords.flatten())
    target_xyz = source_proj.transform_points(target_proj, target_x_points.flatten(), target_y_points.flatten())
    indices = np.zeros(target_xyz.shape[0], dtype=int)
    finite_xyz = np.all(np.isfinite(target_xyz), axis=-1)
    if _is_pykdtree:
        kdtree = _kdtreeClass(xyz)
        _, indices[finite_xyz] = kdtree.query(target_xyz[finite_xyz, :], k=1, sqr_dists=True)
    else:
        kdtree = _kdtreeClass(xyz, balanced_tree=False)
        _, indices[finite_xyz] = kdtree.query(target_xyz[finite_xyz, :], k=1)
    mask = ~finite_xyz | (indices >= len(xyz))
    indices[mask] = 0
    desired_ny, desired_nx = target_x_points.shape
    temp_array = array.reshape((-1,) + array.shape[2:])
    if np.any(mask):
        new_array = np.ma.array(temp_array[indices])
        new_array[mask] = np.ma.masked
    else:
        new_array = temp_array[indices]
    new_array.shape = (desired_ny, desired_nx) + array.shape[2:]
    back_to_target_xyz = target_proj.transform_points(source_proj, target_xyz[:, 0], target_xyz[:, 1])
    back_to_target_x = back_to_target_xyz[:, 0].reshape(desired_ny, desired_nx)
    back_to_target_y = back_to_target_xyz[:, 1].reshape(desired_ny, desired_nx)
    FRACTIONAL_OFFSET_THRESHOLD = 0.1
    x_extent = np.abs(target_proj.x_limits[1] - target_proj.x_limits[0])
    y_extent = np.abs(target_proj.y_limits[1] - target_proj.y_limits[0])
    non_self_inverse_points = (np.abs(target_x_points - back_to_target_x) / x_extent > FRACTIONAL_OFFSET_THRESHOLD) | (np.abs(target_y_points - back_to_target_y) / y_extent > FRACTIONAL_OFFSET_THRESHOLD)
    if np.any(non_self_inverse_points):
        if not np.ma.isMaskedArray(new_array):
            new_array = np.ma.array(new_array, mask=False)
        new_array[non_self_inverse_points] = np.ma.masked
    if mask_extrapolated:
        target_in_source_x = target_xyz[:, 0].reshape(desired_ny, desired_nx)
        target_in_source_y = target_xyz[:, 1].reshape(desired_ny, desired_nx)
        bounds = _determine_bounds(source_x_coords, source_y_coords, source_proj)
        outside_source_domain = (target_in_source_y >= bounds['y'][1]) | (target_in_source_y <= bounds['y'][0])
        tmp_inside = np.zeros_like(outside_source_domain)
        for bound_x in bounds['x']:
            tmp_inside = tmp_inside | (target_in_source_x <= bound_x[1]) & (target_in_source_x >= bound_x[0])
        outside_source_domain = outside_source_domain | ~tmp_inside
        if np.any(outside_source_domain):
            if not np.ma.isMaskedArray(new_array):
                new_array = np.ma.array(new_array, mask=False)
            new_array[outside_source_domain] = np.ma.masked
    return new_array