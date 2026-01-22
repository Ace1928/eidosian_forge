import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def fetch_geometries(self, projection, extent):
    """
        Return any Point, Linestring or LinearRing geometries available
        from the WFS that lie within the specified extent.

        Parameters
        ----------
        projection: :class:`cartopy.crs.Projection`
            The projection in which the extent is specified and in
            which the geometries should be returned. Only the default
            (native) projection is supported.
        extent: four element tuple
            (min_x, max_x, min_y, max_y) tuple defining the geographic extent
            of the geometries to obtain.

        Returns
        -------
        geoms
            A list of Shapely geometries.

        """
    if self.default_projection() != projection:
        raise ValueError(f'Geometries are only available in projection {self.default_projection()!r}.')
    min_x, max_x, min_y, max_y = extent
    response = self.service.getfeature(typename=self.features, bbox=(min_x, min_y, max_x, max_y), **self.getfeature_extra_kwargs)
    geoms_by_srs = self._to_shapely_geoms(response)
    if not geoms_by_srs:
        geoms = []
    elif len(geoms_by_srs) > 1:
        raise ValueError('Unexpected response from the WFS server. The geometries are in multiple SRSs, when only one was expected.')
    else:
        srs, geoms = list(geoms_by_srs.items())[0]
        if srs is not None:
            if srs in _URN_TO_CRS:
                geom_proj = _URN_TO_CRS[srs]
                if geom_proj != projection:
                    raise ValueError(f'The geometries are not in expected projection. Expected {projection!r}, got {geom_proj!r}.')
            elif ':EPSG:' in srs:
                epsg_num = srs.split(':')[-1]
                geom_proj = ccrs.epsg(int(epsg_num))
                if geom_proj != projection:
                    raise ValueError(f'The EPSG geometries are not in expected  projection. Expected {projection!r},  got {geom_proj!r}.')
            else:
                warnings.warn(f'Unable to verify matching projections due to incomplete mappings from SRS identifiers to cartopy projections. The geometries have an SRS of {srs!r}.')
    return geoms