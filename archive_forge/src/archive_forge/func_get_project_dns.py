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
def get_project_dns(dns_access, credential_data):
    with NamedTemporaryFile() as credfile:
        credfile.write(json.dumps(credential_data).encode('utf-8'))
        credfile.flush()
        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': credfile.name}):
            return dns_access()