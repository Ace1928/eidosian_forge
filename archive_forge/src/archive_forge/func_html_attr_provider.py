from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
@pytest.fixture
def html_attr_provider():
    return TileProvider(url='https://myserver.com/tiles/{z}/{x}/{y}.png', attribution='(C) xyzservices', html_attribution='&copy; <a href="https://xyzservices.readthedocs.io">xyzservices</a>', name='my_public_provider_html')