import datetime
import json
import os
from six.moves import http_client
from six.moves.urllib import parse as urlparse
from oauth2client import _helpers
from oauth2client import client
from oauth2client import transport
def get_service_account_info(http, service_account='default'):
    """Get information about a service account from the metadata server.

    Args:
        http: an object to be used to make HTTP requests.
        service_account: An email specifying the service account for which to
            look up information. Default will be information for the "default"
            service account of the current compute engine instance.

    Returns:
         A dictionary with information about the specified service account,
         for example:

            {
                'email': '...',
                'scopes': ['scope', ...],
                'aliases': ['default', '...']
            }
    """
    return get(http, 'instance/service-accounts/{0}/'.format(service_account), recursive=True)