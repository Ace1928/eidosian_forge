from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
def auth_settings(self):
    """
        Gets Auth Settings dict for api client.

        :return: The Auth Settings information dict.
        """
    return {'BearerToken': {'type': 'api_key', 'in': 'header', 'key': 'authorization', 'value': self.get_api_key_with_prefix('authorization')}}