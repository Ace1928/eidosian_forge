from xml.etree.ElementTree import ParseError
import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.skipif(not _HAS_PYKDTREE_OR_SCIPY, reason='pykdtree or scipy is required')
@pytest.mark.mpl_image_compare(filename='gshhs_coastlines.png', tolerance=0.95)
def test_gshhs():
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.set_extent([138, 142, 32, 42], ccrs.Geodetic())
    ax.stock_img()
    ax.add_feature(cfeature.GSHHSFeature('coarse', edgecolor='red'))
    ax.add_feature(cfeature.GSHHSFeature('low', levels=[2], facecolor='green'), facecolor='blue')
    return ax.figure