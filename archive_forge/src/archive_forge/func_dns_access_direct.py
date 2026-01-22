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
def dns_access_direct(request, project_id):
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform.read-only'], request=request)
    headers = {}
    credentials.apply(headers)
    response = request(url='https://dns.googleapis.com/dns/v1/projects/{}'.format(project_id), headers=headers)
    if response.status == 200:
        return response.data