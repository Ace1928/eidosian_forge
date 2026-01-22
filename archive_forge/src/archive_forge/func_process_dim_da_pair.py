import warnings
import numpy as np
import xarray as xr
def process_dim_da_pair(self, da, dim_sublist):
    da_dims = da.dims
    out_dims = [dim for dim in da_dims if dim in self.potential_out_dims and dim not in dim_sublist]
    subscripts = ''
    updated_in_dims = dim_sublist.copy()
    for dim in out_dims:
        self.out_dims.append(dim)
        sub = self.einsum_axes.pop()
        self.out_subscript += sub
        subscripts += sub
        updated_in_dims.insert(0, dim)
    for dim in dim_sublist:
        subscripts += self.dim_map[dim]
    if len(da_dims) > len(out_dims) + len(dim_sublist):
        return (f'...{subscripts}', updated_in_dims)
    return (subscripts, updated_in_dims)