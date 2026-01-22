import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
def srtm_fname(self, lon, lat):
    """
        Return the filename for the given lon/lat SRTM tile (downloading if
        necessary), or None if no such tile exists (i.e. the tile would be
        entirely over water, or out of latitude range).

        """
    if int(lon) != lon or int(lat) != lat:
        raise ValueError('Integer longitude/latitude values required.')
    x = '%s%03d' % ('E' if lon >= 0 else 'W', abs(int(lon)))
    y = '%s%02d' % ('N' if lat >= 0 else 'S', abs(int(lat)))
    srtm_downloader = Downloader.from_config(('SRTM', f'SRTM{self._resolution}'))
    params = {'config': config, 'resolution': self._resolution, 'x': x, 'y': y}
    if srtm_downloader.url(params) is None:
        return None
    else:
        return self.downloader.path(params)