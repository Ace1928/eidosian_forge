import logging
import time
import requests
from oslo_utils import importutils
from troveclient.apiclient import exceptions
@staticmethod
def concat_url(endpoint, url):
    """Concatenate endpoint and final URL.

        E.g., "http://keystone/v2.0/" and "/tokens" are concatenated to
        "http://keystone/v2.0/tokens".

        :param endpoint: the base URL
        :param url: the final URL
        """
    return '%s/%s' % (endpoint.rstrip('/'), url.strip('/'))