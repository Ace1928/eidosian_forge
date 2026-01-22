import param
import numpy as np
from bokeh.models import MercatorTileSource
from cartopy import crs as ccrs
from cartopy.feature import Feature as cFeature
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.shapereader import Reader
from holoviews.core import Element2D, Dimension, Dataset as HvDataset, NdOverlay, Overlay
from holoviews.core import util
from holoviews.element import (
from holoviews.element.selection import Selection2DExpr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import (
from shapely.ops import unary_union
from ..util import (
def trimesh(self):
    trimesh = super().trimesh()
    node_params = util.get_param_values(trimesh.nodes)
    node_params['crs'] = self.crs
    nodes = TriMesh.node_type(trimesh.nodes.data, **node_params)
    return TriMesh((trimesh.data, nodes), crs=self.crs, **util.get_param_values(trimesh))