from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
def get_basic_auth_token(self):
    """
        Gets HTTP basic authentication header (string).

        :return: The token for basic HTTP authentication.
        """
    return urllib3.util.make_headers(basic_auth=self.username + ':' + self.password).get('authorization')