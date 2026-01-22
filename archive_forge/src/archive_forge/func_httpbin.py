import ssl
import tempfile
import threading
import pytest
from requests.compat import urljoin
@pytest.fixture
def httpbin(httpbin):
    return prepare_url(httpbin)