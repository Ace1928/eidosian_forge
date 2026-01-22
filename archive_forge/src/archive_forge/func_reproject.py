from math import ceil, floor
from affine import Affine
import numpy as np
import rasterio
from rasterio._base import _transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.env import ensure_env, require_gdal_version
from rasterio.errors import TransformError, RPCError
from rasterio.transform import array_bounds
from rasterio._warp import (
@ensure_env
def reproject(source, destination=None, src_transform=None, gcps=None, rpcs=None, src_crs=None, src_nodata=None, dst_transform=None, dst_crs=None, dst_nodata=None, dst_resolution=None, src_alpha=0, dst_alpha=0, resampling=Resampling.nearest, num_threads=1, init_dest_nodata=True, warp_mem_limit=0, **kwargs):
    """Reproject a source raster to a destination raster.

    If the source and destination are ndarrays, coordinate reference
    system definitions and affine transformation parameters or ground
    control points (gcps) are required for reprojection.

    If the source and destination are rasterio Bands, shorthand for
    bands of datasets on disk, the coordinate reference systems and
    transforms or GCPs will be read from the appropriate datasets.

    Parameters
    ------------
    source: ndarray or Band
        The source is a 2 or 3-D ndarray, or a single or a multiple
        Rasterio Band object. The dimensionality of source
        and destination must match, i.e., for multiband reprojection
        the lengths of the first axes of the source and destination
        must be the same.
    destination: ndarray or Band, optional
        The destination is a 2 or 3-D ndarray, or a single or a multiple
        Rasterio Band object. The dimensionality of source
        and destination must match, i.e., for multiband reprojection
        the lengths of the first axes of the source and destination
        must be the same.
    src_transform: affine.Affine(), optional
        Source affine transformation. Required if source and
        destination are ndarrays. Will be derived from source if it is
        a rasterio Band. An error will be raised if this parameter is
        defined together with gcps.
    gcps: sequence of GroundControlPoint, optional
        Ground control points for the source. An error will be raised
        if this parameter is defined together with src_transform or rpcs.
    rpcs: RPC or dict, optional
        Rational polynomial coefficients for the source. An error will
        be raised if this parameter is defined together with src_transform
        or gcps.
    src_crs: CRS or dict, optional
        Source coordinate reference system, in rasterio dict format.
        Required if source and destination are ndarrays.
        Will be derived from source if it is a rasterio Band.
        Example: CRS({'init': 'EPSG:4326'})
    src_nodata: int or float, optional
        The source nodata value. Pixels with this value will not be
        used for interpolation. If not set, it will default to the
        nodata value of the source image if a masked ndarray or
        rasterio band, if available.
    dst_transform: affine.Affine(), optional
        Target affine transformation. Required if source and
        destination are ndarrays. Will be derived from target if it is
        a rasterio Band.
    dst_crs: CRS or dict, optional
        Target coordinate reference system. Required if source and
        destination are ndarrays. Will be derived from target if it
        is a rasterio Band.
    dst_nodata: int or float, optional
        The nodata value used to initialize the destination; it will
        remain in all areas not covered by the reprojected source.
        Defaults to the nodata value of the destination image (if set),
        the value of src_nodata, or 0 (GDAL default).
    dst_resolution: tuple (x resolution, y resolution) or float, optional
        Target resolution, in units of target coordinate reference
        system.
    src_alpha : int, optional
        Index of a band to use as the alpha band when warping.
    dst_alpha : int, optional
        Index of a band to use as the alpha band when warping.
    resampling: int, rasterio.enums.Resampling
        Resampling method to use.
        Default is :attr:`rasterio.enums.Resampling.nearest`.
        An exception will be raised for a method not supported by the running
        version of GDAL.
    num_threads : int, optional
        The number of warp worker threads. Default: 1.
    init_dest_nodata: bool
        Flag to specify initialization of nodata in destination;
        prevents overwrite of previous warps. Defaults to True.
    warp_mem_limit : int, optional
        The warp operation memory limit in MB. Larger values allow the
        warp operation to be carried out in fewer chunks. The amount of
        memory required to warp a 3-band uint8 2000 row x 2000 col
        raster to a destination of the same size is approximately
        56 MB. The default (0) means 64 MB with GDAL 2.2.
    kwargs:  dict, optional
        Additional arguments passed to both the image to image
        transformer :cpp:func:`GDALCreateGenImgProjTransformer2` (for example,
        MAX_GCP_ORDER=2) and the :cpp:struct:`GDALWarpOptions` (for example,
        INIT_DEST=NO_DATA).

    Returns
    ---------
    destination: ndarray or Band
        The transformed narray or Band.
    dst_transform: Affine
        THe affine transformation matrix of the destination.
    """
    if src_transform and gcps or (src_transform and rpcs) or (gcps and rpcs):
        raise ValueError('src_transform, gcps, and rpcs are mutually exclusive parameters and may not be used together.')
    try:
        if resampling == 7:
            raise ValueError('Gauss resampling is not supported')
        Resampling(resampling)
    except ValueError:
        raise ValueError('resampling must be one of: {0}'.format(', '.join(['Resampling.{0}'.format(r.name) for r in SUPPORTED_RESAMPLING])))
    if destination is None and dst_transform is not None:
        raise ValueError('Must provide destination if dst_transform is provided.')
    if dst_transform is None and (destination is None or isinstance(destination, np.ndarray)):
        src_bounds = tuple([None] * 4)
        if isinstance(source, np.ndarray):
            if source.ndim == 3:
                src_count, src_height, src_width = source.shape
            else:
                src_count = 1
                src_height, src_width = source.shape
            if not (gcps or rpcs):
                src_bounds = array_bounds(src_height, src_width, src_transform)
        else:
            src_rdr, src_bidx, _, src_shape = source
            if not (src_rdr.transform.is_identity and src_rdr.crs is None):
                src_bounds = src_rdr.bounds
            src_crs = src_crs or src_rdr.crs
            if rpcs:
                if isinstance(src_crs, str):
                    src_crs_obj = rasterio.crs.CRS.from_string(src_crs)
                else:
                    src_crs_obj = src_crs
                if src_crs is not None and src_crs_obj.to_epsg() != 4326:
                    raise RPCError('Reprojecting with rational polynomial coefficients using source CRS other than EPSG:4326')
            if isinstance(src_bidx, int):
                src_bidx = [src_bidx]
            src_count = len(src_bidx)
            src_height, src_width = src_shape
            gcps = src_rdr.gcps[0] if src_rdr.gcps[0] and (not rpcs) else None
        dst_height = None
        dst_width = None
        dst_count = src_count
        if isinstance(destination, np.ndarray):
            if destination.ndim == 3:
                dst_count, dst_height, dst_width = destination.shape
            else:
                dst_count = 1
                dst_height, dst_width = destination.shape
        left, bottom, right, top = src_bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(src_crs=src_crs, dst_crs=dst_crs, width=src_width, height=src_height, left=left, bottom=bottom, right=right, top=top, gcps=gcps, rpcs=rpcs, dst_width=dst_width, dst_height=dst_height, resolution=dst_resolution)
        if destination is None:
            destination = np.empty((int(dst_count), int(dst_height), int(dst_width)), dtype=source.dtype)
    _reproject(source, destination, src_transform=src_transform, gcps=gcps, rpcs=rpcs, src_crs=src_crs, src_nodata=src_nodata, dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=dst_nodata, dst_alpha=dst_alpha, src_alpha=src_alpha, resampling=resampling, init_dest_nodata=init_dest_nodata, num_threads=num_threads, warp_mem_limit=warp_mem_limit, **kwargs)
    return (destination, dst_transform)