from __future__ import (absolute_import, division, print_function)
import traceback
import os
import ssl
import time
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def download_disk_image(connection, module):

    def _transfer(transfer_service, proxy_connection, proxy_url, transfer_ticket):
        BUF_SIZE = 128 * 1024
        transfer_headers = {'Authorization': transfer_ticket}
        proxy_connection.request('GET', proxy_url.path, headers=transfer_headers)
        r = proxy_connection.getresponse()
        path = module.params['download_image_path']
        image_size = int(r.getheader('Content-Length'))
        with open(path, 'wb') as mydisk:
            pos = 0
            while pos < image_size:
                to_read = min(image_size - pos, BUF_SIZE)
                chunk = r.read(to_read)
                if not chunk:
                    raise RuntimeError('Socket disconnected')
                mydisk.write(chunk)
                pos += len(chunk)
    return transfer(connection, module, otypes.ImageTransferDirection.DOWNLOAD, transfer_func=_transfer)