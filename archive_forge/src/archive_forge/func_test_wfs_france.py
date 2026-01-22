from xml.etree.ElementTree import ParseError
import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
@pytest.mark.network
@pytest.mark.skipif(not _HAS_PYKDTREE_OR_SCIPY or not _OWSLIB_AVAILABLE, reason='OWSLib and at least one of pykdtree or scipy is required')
@pytest.mark.xfail(raises=(ParseError, AttributeError), reason='Bad XML returned from the URL')
@pytest.mark.mpl_image_compare(filename='wfs_france.png')
def test_wfs_france():
    ax = plt.axes(projection=ccrs.epsg(2154))
    url = 'https://agroenvgeo.data.inra.fr:443/geoserver/wfs'
    typename = 'collectif:t_ser_l93'
    feature = cfeature.WFSFeature(url, typename, edgecolor='red')
    ax.add_feature(feature)
    return ax.figure