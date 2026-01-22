import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
def augmented_raise_for_status(response):
    """Wrap the standard `requests.response.raise_for_status()` method and return reason"""
    try:
        response.raise_for_status()
    except HTTPError as e:
        if response.text:
            raise HTTPError(f'{e}. Response text: {response.text}', request=e.request, response=e.response)
        else:
            raise e