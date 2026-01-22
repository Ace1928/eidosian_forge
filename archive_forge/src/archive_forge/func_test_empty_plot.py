from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
def test_empty_plot():
    """Test making a plot with empty arrays."""
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mercator())
    ax.plot([], [], transform=ccrs.PlateCarree())
    fig.savefig(BytesIO())