import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
@pytest.fixture
def srtm_login_or_skip(monkeypatch):
    import os
    try:
        srtm_username = os.environ['SRTM_USERNAME']
    except KeyError:
        pytest.skip('SRTM_USERNAME environment variable is unset.')
    try:
        srtm_password = os.environ['SRTM_PASSWORD']
    except KeyError:
        pytest.skip('SRTM_PASSWORD environment variable is unset.')
    from http.cookiejar import CookieJar
    from urllib.request import HTTPBasicAuthHandler, HTTPCookieProcessor, HTTPPasswordMgrWithDefaultRealm, build_opener
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, 'https://urs.earthdata.nasa.gov', srtm_username, srtm_password)
    cookie_jar = CookieJar()
    opener = build_opener(HTTPBasicAuthHandler(password_manager), HTTPCookieProcessor(cookie_jar))
    monkeypatch.setattr(cartopy.io, 'urlopen', opener.open)