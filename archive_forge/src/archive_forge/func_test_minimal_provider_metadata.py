import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.parametrize('provider_name', xyz.flatten())
def test_minimal_provider_metadata(provider_name):
    provider = xyz.flatten()[provider_name]
    check_provider(provider)