from datashader.compiler import compile_components
from datashader.utils import Dispatcher
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import (
from datashader.utils import apply
import dask
import numpy as np
import xarray as xr
from dask.base import tokenize, compute
from dask.array.overlap import overlap
def dask_curvilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]
    x_coord_name = glyph.x
    y_coord_name = glyph.y
    z_name = glyph.name
    data_dim_names = list(xr_ds[z_name].dims)
    x_coord_dim_names = list(xr_ds[x_coord_name].dims)
    y_coord_dim_names = list(xr_ds[y_coord_name].dims)
    zs = xr_ds[z_name].data
    x_centers = xr_ds[glyph.x].data
    y_centers = xr_ds[glyph.y].data
    var_name = list(xr_ds.data_vars.keys())[0]
    err_msg = 'DataArray {name} is backed by a Dask array, \nbut coordinate {coord} is not backed by a Dask array with identical \ndimension order and chunks'
    if not isinstance(x_centers, dask.array.Array) or xr_ds[glyph.name].dims != xr_ds[glyph.x].dims or xr_ds[glyph.name].chunks != xr_ds[glyph.x].chunks:
        raise ValueError(err_msg.format(name=glyph.name, coord=glyph.x))
    if not isinstance(y_centers, dask.array.Array) or xr_ds[glyph.name].dims != xr_ds[glyph.y].dims or xr_ds[glyph.name].chunks != xr_ds[glyph.y].chunks:
        raise ValueError(err_msg.format(name=glyph.name, coord=glyph.y))
    if x_centers.dtype.kind != 'f':
        x_centers = x_centers.astype(np.float64)
    if y_centers.dtype.kind != 'f':
        y_centers = y_centers.astype(np.float64)
    x_overlapped_centers = overlap(x_centers, depth=1, boundary=np.nan)
    y_overlapped_centers = overlap(y_centers, depth=1, boundary=np.nan)

    def chunk(np_zs, np_x_centers, np_y_centers):
        for centers in [np_x_centers, np_y_centers]:
            if np.isnan(centers[0, :]).all():
                centers[0, :] = centers[1, :] - (centers[2, :] - centers[1, :])
            if np.isnan(centers[-1, :]).all():
                centers[-1, :] = centers[-2, :] + (centers[-2, :] - centers[-3, :])
            if np.isnan(centers[:, 0]).all():
                centers[:, 0] = centers[:, 1] - (centers[:, 2] - centers[:, 1])
            if np.isnan(centers[:, -1]).all():
                centers[:, -1] = centers[:, -2] + (centers[:, -2] - centers[:, -3])
        x_breaks_chunk = glyph.infer_interval_breaks(np_x_centers)
        y_breaks_chunk = glyph.infer_interval_breaks(np_y_centers)
        x_breaks_chunk = x_breaks_chunk[1:-1, 1:-1]
        y_breaks_chunk = y_breaks_chunk[1:-1, 1:-1]
        chunk_coords = {x_coord_name: (x_coord_dim_names, np_x_centers[1:-1, 1:-1]), y_coord_name: (y_coord_dim_names, np_y_centers[1:-1, 1:-1])}
        chunk_ds = xr.DataArray(np_zs, coords=chunk_coords, dims=data_dim_names, name=var_name).to_dataset()
        aggs = create(shape)
        extend(aggs, chunk_ds, st, bounds, x_breaks=x_breaks_chunk, y_breaks=y_breaks_chunk)
        return aggs
    result_name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    z_keys = [k for row in zs.__dask_keys__() for k in row]
    x_overlap_keys = [k for row in x_overlapped_centers.__dask_keys__() for k in row]
    y_overlap_keys = [k for row in y_overlapped_centers.__dask_keys__() for k in row]
    result_keys = [(result_name, i) for i in range(len(z_keys))]
    dsk = dict(((res_k, (chunk, z_k, x_k, y_k)) for res_k, z_k, x_k, y_k in zip(result_keys, z_keys, x_overlap_keys, y_overlap_keys)))
    dsk[result_name] = (apply, finalize, [(combine, result_keys)], dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label], attrs=dict(x_range=x_range, y_range=y_range)))
    dsk.update(x_overlapped_centers.dask)
    dsk.update(y_overlapped_centers.dask)
    return (dsk, result_name)