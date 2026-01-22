import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
@property
def geometry(self):
    """
        A shapely.geometry instance for this Record.

        The geometry may be ``None`` if a null shape is defined in the
        shapefile.

        """
    if not self._geometry and self._shape.shapeType != shapefile.NULL:
        self._geometry = sgeom.shape(self._shape)
    return self._geometry