import datetime
import email.utils
import hashlib
import logging
import random
import time
from urllib import parse
from oslo_config import cfg
from swiftclient import client as sc
from swiftclient import exceptions
from swiftclient import utils as swiftclient_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
def get_files_from_container(self, files_container, files_to_skip=None):
    """Gets the file contents from a container.

         Get the file contents from the container in a files map. A list
         of files to skip can also be specified and those would not be
         downloaded from swift.
         """
    client = self.client()
    files = {}
    if files_to_skip is None:
        files_to_skip = []
    try:
        headers, objects = client.get_container(files_container)
        bytes_used = int(headers.get('x-container-bytes-used', 0))
        if bytes_used > cfg.CONF.max_json_body_size:
            msg = _('Total size of files to download (%(size)s bytes) exceeds maximum allowed (%(limit)s bytes).') % {'size': bytes_used, 'limit': cfg.CONF.max_json_body_size}
            raise exception.DownloadLimitExceeded(message=msg)
        for obj in objects:
            file_name = obj['name']
            if file_name not in files_to_skip:
                contents = client.get_object(files_container, file_name)[1]
            files[file_name] = contents
    except exceptions.ClientException as cex:
        raise exception.NotFound(_('Could not fetch files from container %(container)s, reason: %(reason)s.') % {'container': files_container, 'reason': str(cex)})
    return files