import datetime
import http.client as http_client
import json
import logging
import os
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import metrics
def is_on_gce(request):
    """Checks to see if the code runs on Google Compute Engine

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.

    Returns:
        bool: True if the code runs on Google Compute Engine, False otherwise.
    """
    if ping(request):
        return True
    if os.name == 'nt':
        return False
    return detect_gce_residency_linux()