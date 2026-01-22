import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
def read_SRTM(fh):
    """
    Read the array of (y, x) elevation data from the given named file-handle.

    Parameters
    ----------
    fh
        A named file-like as passed through to :func:`cartopy.io.fh_getter`.
        The filename is used to determine the extent of the resulting array.

    Returns
    -------
    elevation
        The elevation values from the SRTM file. Data is flipped
        vertically such that the higher the y-index, the further north the
        data. Data shape is automatically determined by the size of data read
        from file, and is either (1201, 1201) for 3 arc-second data or
        (3601, 3601) for 1 arc-second data.
    crs: :class:`cartopy.crs.CRS`
        The coordinate reference system of the extents.
    extents: 4-tuple (x0, x1, y0, y1)
        The boundaries of the returned elevation array.

    """
    fh, fname = fh_getter(fh, needs_filename=True)
    if fname.endswith('.zip'):
        from zipfile import ZipFile
        zfh = ZipFile(fh, 'rb')
        fh = zfh.open(Path(fname).stem, 'r')
    elev = np.fromfile(fh, dtype=np.dtype('>i2'))
    if elev.size == 12967201:
        elev.shape = (3601, 3601)
    elif elev.size == 1442401:
        elev.shape = (1201, 1201)
    else:
        raise ValueError(f'Shape of SRTM data ({elev.size}) is unexpected.')
    fname = Path(fname).name
    y_dir, y, x_dir, x = (fname[0], int(fname[1:3]), fname[3], int(fname[4:7]))
    if y_dir == 'S':
        y *= -1
    if x_dir == 'W':
        x *= -1
    return (elev[::-1, ...], ccrs.PlateCarree(), (x, x + 1, y, y + 1))