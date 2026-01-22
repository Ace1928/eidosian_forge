import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
@staticmethod
def default_downloader():
    """
        Return a GSHHSShpDownloader instance that expects (and if necessary
        downloads and installs) shapefiles in the data directory of the
        cartopy installation.

        Typically, a user will not need to call this staticmethod.

        To find the path template of the GSHHSShpDownloader:

            >>> gshhs_dnldr = GSHHSShpDownloader.default_downloader()
            >>> print(gshhs_dnldr.target_path_template)
            {config[data_dir]}/shapefiles/gshhs/{scale}/GSHHS_{scale}_L{level}.shp

        """
    default_spec = ('shapefiles', 'gshhs', '{scale}', 'GSHHS_{scale}_L{level}.shp')
    gshhs_path_template = str(Path('{config[data_dir]}').joinpath(*default_spec))
    pre_path_tmplt = str(Path('{config[pre_existing_data_dir]}').joinpath(*default_spec))
    return GSHHSShpDownloader(target_path_template=gshhs_path_template, pre_downloaded_path_template=pre_path_tmplt)