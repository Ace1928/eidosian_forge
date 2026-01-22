from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def create_update(self, rest_path, data):
    """
        Create or Update a file/directory monitor data input in Splunk
        """
    if data is not None and self.override:
        data = self.get_urlencoded_data(data)
    return self.post('/{0}?output_mode=json'.format(rest_path), payload=data)