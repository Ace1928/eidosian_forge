from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
def test_requires_token(private_provider, basic_provider):
    assert private_provider.requires_token() is True
    assert basic_provider.requires_token() is False