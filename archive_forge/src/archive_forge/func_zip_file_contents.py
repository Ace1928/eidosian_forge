import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
def zip_file_contents(self, format_dict):
    """
        Return a generator of the filenames to be found in the downloaded
        GSHHS zip file for the specified resource.

        """
    for ext in ['.shp', '.dbf', '.shx']:
        p = Path('GSHHS_shp', '{scale}', 'GSHHS_{scale}_L{level}{extension}')
        yield str(p).format(extension=ext, **format_dict)