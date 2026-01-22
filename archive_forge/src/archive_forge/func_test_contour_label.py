import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='contour_label.png', tolerance=3.9 if _MPL_38 else 0.5)
def test_contour_label():
    from cartopy.tests.mpl.test_caching import sample_data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())
    ax.set_global()
    ax.coastlines('110m', alpha=0.1)
    x, y, z = sample_data((20, 40))
    z = z * -1.5 * y
    filled_c = ax.contourf(x, y, z, transform=ccrs.PlateCarree())
    line_c = ax.contour(x, y, z, levels=filled_c.levels, colors=['black'], transform=ccrs.PlateCarree())
    fig.colorbar(filled_c, orientation='horizontal')
    ax.clabel(line_c, colors=['black'], manual=False, inline=True, fmt=' {:.0f} '.format)
    return fig