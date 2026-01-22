import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.HEREv3)
def test_herev3(provider_name):
    try:
        token = os.environ['HEREV3']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.HEREv3[provider_name](apiKey=token)
    get_test_result(provider, allow_403=False)