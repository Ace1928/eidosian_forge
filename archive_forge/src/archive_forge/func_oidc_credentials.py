import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
@pytest.fixture
def oidc_credentials(service_account_file, http_request):
    result = service_account.IDTokenCredentials.from_service_account_file(service_account_file, target_audience=_AUDIENCE_OIDC)
    result.refresh(http_request)
    yield result