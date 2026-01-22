from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def create_transfer_connection(module, transfer, context, connect_timeout=10, read_timeout=60):
    url = urlparse(transfer.transfer_url)
    connection = HTTPSConnection(url.netloc, context=context, timeout=connect_timeout)
    try:
        connection.connect()
    except Exception as e:
        module.warn('Cannot connect to %s, trying %s: %s' % (transfer.transfer_url, transfer.proxy_url, e))
        url = urlparse(transfer.proxy_url)
        connection = HTTPSConnection(url.netloc, context=context, timeout=connect_timeout)
        connection.connect()
    connection.sock.settimeout(read_timeout)
    return (connection, url)