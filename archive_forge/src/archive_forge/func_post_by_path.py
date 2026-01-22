from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from copy import copy
import json
def post_by_path(self, rest_path, data=None):
    """
        POST with data to path
        """
    if data is None:
        data = json.dumps(self.get_data())
    elif data is False:
        return self.post('/{0}'.format(rest_path))
    return self.post('/{0}'.format(rest_path), payload=data)