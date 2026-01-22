import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.Stadia)
def test_stadia(provider_name):
    try:
        token = os.environ['STADIA']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.Stadia[provider_name](api_key=token)
    provider['url'] = provider['url'] + '?api_key={api_key}'
    get_test_result(provider, allow_403=False)