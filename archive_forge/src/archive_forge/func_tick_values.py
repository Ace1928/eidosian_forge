import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def tick_values(self, vmin, vmax):
    vmin = max(vmin, -90.0)
    vmax = min(vmax, 90.0)
    return LongitudeLocator.tick_values(self, vmin, vmax)