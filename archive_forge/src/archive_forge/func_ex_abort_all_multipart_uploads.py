import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def ex_abort_all_multipart_uploads(self, container, prefix=None):
    """
        Extension method for removing all partially completed OSS multipart
        uploads.

        :param container: The container holding the uploads
        :type container: :class:`Container`

        :keyword prefix: Delete only uploads of objects with this prefix
        :type prefix: ``str``
        """
    for upload in self.ex_iterate_multipart_uploads(container, prefix, delimiter=None):
        object_path = self._get_object_path(container, upload.key)
        self._abort_multipart(object_path, upload.id, container=container)