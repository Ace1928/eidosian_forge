import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('name', flat_free)
def test_free_providers(name):
    provider = flat_free[name]
    if 'Stadia' in name:
        pytest.skip("Stadia doesn't support tile download in this way.")
    get_test_result(provider)