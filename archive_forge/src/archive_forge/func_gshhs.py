import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
def gshhs(scale='c', level=1):
    """
    Return the path to the requested GSHHS shapefile,
    downloading and unzipping if necessary.

    """
    gshhs_downloader = Downloader.from_config(('shapefiles', 'gshhs', scale, level))
    format_dict = {'config': config, 'scale': scale, 'level': level}
    return gshhs_downloader.path(format_dict)