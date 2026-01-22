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
def dask_raster(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]
    chunk_inds = {}
    for k, chunks in xr_ds.chunks.items():
        chunk_inds[k] = [0] + list(np.cumsum(chunks))
    x_name = glyph.x
    y_name = glyph.y
    coords = xr_ds[glyph.name].coords
    coord_dims = list(coords.dims)
    xdim_ind = coord_dims.index(x_name)
    ydim_ind = coord_dims.index(y_name)
    var_name = list(xr_ds.data_vars.keys())[0]
    src_x0, src_x1 = glyph._compute_bounds_from_1d_centers(xr_ds, x_name, maybe_expand=False, orient=False)
    src_y0, src_y1 = glyph._compute_bounds_from_1d_centers(xr_ds, y_name, maybe_expand=False, orient=False)
    xbinsize = float(xr_ds[x_name][1] - xr_ds[x_name][0])
    ybinsize = float(xr_ds[y_name][1] - xr_ds[y_name][0])
    out_h, out_w = shape
    src_h, src_w = [xr_ds[glyph.name].shape[i] for i in [ydim_ind, xdim_ind]]
    out_x0, out_x1, out_y0, out_y1 = bounds
    scale_y, translate_y = build_scale_translate(out_h, out_y0, out_y1, src_h, src_y0, src_y1)
    scale_x, translate_x = build_scale_translate(out_w, out_x0, out_x1, src_w, src_x0, src_x1)

    def chunk(np_arr, *inds):
        chunk_coords_list = []
        for i, coord_name in enumerate(coords.dims):
            chunk_number = inds[i]
            coord_slice = slice(chunk_inds[coord_name][chunk_number], chunk_inds[coord_name][chunk_number + 1])
            chunk_coords_list.append([coord_name, coords[coord_name][coord_slice]])
        chunk_coords = dict(chunk_coords_list)
        chunk_ds = xr.DataArray(np_arr, coords=chunk_coords, dims=coord_dims, name=var_name).to_dataset()
        x_chunk_number = inds[xdim_ind]
        offset_x = chunk_inds[x_name][x_chunk_number]
        y_chunk_number = inds[ydim_ind]
        offset_y = chunk_inds[y_name][y_chunk_number]
        aggs = create(shape)
        extend(aggs, chunk_ds, st, bounds, scale_x=scale_x, scale_y=scale_y, translate_x=translate_x, translate_y=translate_y, offset_x=offset_x, offset_y=offset_y, src_xbinsize=xbinsize, src_ybinsize=ybinsize)
        return aggs
    name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    keys = [k for row in xr_ds.__dask_keys__()[0] for k in row]
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict(((k2, (chunk, k, k[1], k[2])) for k2, k in zip(keys2, keys)))
    dsk[name] = (apply, finalize, [(combine, keys2)], dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label], attrs=dict(x_range=x_range, y_range=y_range)))
    return (dsk, name)