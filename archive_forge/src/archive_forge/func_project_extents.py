import numpy as np
import shapely
import shapely.geometry as sgeom
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from holoviews.element import Tiles
from packaging.version import Version
from shapely.geometry import (
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import transform
from ._warnings import warn
def project_extents(extents, src_proj, dest_proj, tol=1e-06):
    x1, y1, x2, y2 = extents
    if isinstance(src_proj, ccrs.PlateCarree) and (not isinstance(dest_proj, ccrs.PlateCarree)) and (src_proj.proj4_params['lon_0'] != 0):
        xoffset = src_proj.proj4_params['lon_0']
        x1 = x1 - xoffset
        x2 = x2 - xoffset
        src_proj = ccrs.PlateCarree()
    cy1, cy2 = src_proj.y_limits
    if y1 < cy1:
        y1 = cy1
    if y2 > cy2:
        y2 = cy2
    x1 += tol
    x2 -= tol
    y1 += tol
    y2 -= tol
    cx1, cx2 = src_proj.x_limits
    if isinstance(src_proj, ccrs._CylindricalProjection):
        lons = wrap_lons(np.linspace(x1, x2, 10000), -180.0, 360.0)
        x1, x2 = (lons.min(), lons.max())
    else:
        if x1 < cx1:
            x1 = cx1
        if x2 > cx2:
            x2 = cx2
    domain_in_src_proj = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    boundary_poly = Polygon(src_proj.boundary)
    dest_poly = src_proj.project_geometry(Polygon(dest_proj.boundary), dest_proj).buffer(0)
    if src_proj != dest_proj:
        eroded_boundary = boundary_poly.buffer(-src_proj.threshold)
        geom_in_src_proj = eroded_boundary.intersection(domain_in_src_proj)
        try:
            geom_clipped_to_dest_proj = dest_poly.intersection(geom_in_src_proj)
        except Exception:
            geom_clipped_to_dest_proj = None
        if geom_clipped_to_dest_proj:
            geom_in_src_proj = geom_clipped_to_dest_proj
        try:
            geom_in_crs = dest_proj.project_geometry(geom_in_src_proj, src_proj)
        except ValueError as e:
            src_name = type(src_proj).__name__
            dest_name = type(dest_proj).__name__
            raise ValueError(f'Could not project data from {src_name} projection to {dest_name} projection. Ensure the coordinate reference system (crs) matches your data and the kdims.') from e
    else:
        geom_in_crs = boundary_poly.intersection(domain_in_src_proj)
    return geom_in_crs.bounds