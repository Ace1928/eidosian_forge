from matplotlib.path import Path
import numpy as np
import shapely.geometry as sgeom
def not_zero_poly(geom):
    return isinstance(geom, sgeom.Polygon) and (not geom.is_empty) and (geom.area != 0) or not isinstance(geom, sgeom.Polygon)