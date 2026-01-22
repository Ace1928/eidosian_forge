import json
import os
from google.auth import _helpers
import google.auth.transport.requests
import google.auth.transport.urllib3
import pytest
import pytest_asyncio
import requests
import urllib3
import aiohttp
from google.auth.transport import _aiohttp_requests as aiohttp_requests
from system_tests.system_tests_sync import conftest as sync_conftest
@pytest_asyncio.fixture
def service_account_file():
    """The full path to a valid service account key file."""
    yield sync_conftest.SERVICE_ACCOUNT_FILE