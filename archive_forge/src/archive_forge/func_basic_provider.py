from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
@pytest.fixture
def basic_provider():
    return TileProvider(url='https://myserver.com/tiles/{z}/{x}/{y}.png', attribution='(C) xyzservices', name='my_public_provider')