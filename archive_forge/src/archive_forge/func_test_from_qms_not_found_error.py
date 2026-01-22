from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
@pytest.mark.xfail(reason='timeout error', raises=URLError)
def test_from_qms_not_found_error():
    with pytest.raises(ValueError):
        TileProvider.from_qms('LolWut')