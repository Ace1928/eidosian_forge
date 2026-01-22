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
def get_universe_domain(request):
    """Get the universe domain value from the metadata server.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.

    Returns:
        str: The universe domain value. If the universe domain endpoint is not
        not found, return the default value, which is googleapis.com

    Raises:
        google.auth.exceptions.TransportError: if an error other than
            404 occurs while retrieving metadata.
    """
    universe_domain = get(request, 'universe/universe_domain', return_none_for_not_found_error=True)
    if not universe_domain:
        return 'googleapis.com'
    return universe_domain