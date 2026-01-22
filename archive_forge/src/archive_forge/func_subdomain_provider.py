from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
@pytest.fixture
def subdomain_provider():
    return TileProvider(url='https://{s}.myserver.com/tiles/{z}/{x}/{y}.png', attribution='(C) xyzservices', subdomains='abcd', name='my_subdomain_provider')