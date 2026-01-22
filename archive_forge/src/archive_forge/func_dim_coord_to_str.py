from typing import Union
def dim_coord_to_str(self, dim, coord_val, coord_idx):
    """WIP."""
    dim_str = self.dim_map.get(dim, dim)
    coord_str = self.coord_map.get(dim, {}).get(coord_val, coord_val)
    return super().dim_coord_to_str(dim_str, coord_str, coord_idx)