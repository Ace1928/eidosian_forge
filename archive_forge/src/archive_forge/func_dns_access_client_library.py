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
def dns_access_client_library(_, project_id):
    service = discovery.build('dns', 'v1')
    request = service.projects().get(project=project_id)
    return request.execute()