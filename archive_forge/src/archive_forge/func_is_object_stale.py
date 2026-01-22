from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def is_object_stale(self, container, name, filename, file_md5=None, file_sha256=None):
    """Check to see if an object matches the hashes of a file.

        :param container: Name of the container.
        :param name: Name of the object.
        :param filename: Path to the file.
        :param file_md5: Pre-calculated md5 of the file contents. Defaults to
            None which means calculate locally.
        :param file_sha256: Pre-calculated sha256 of the file contents.
            Defaults to None which means calculate locally.
        """
    try:
        metadata = self.get_object_metadata(name, container).metadata
    except exceptions.NotFoundException:
        self._connection.log.debug('swift stale check, no object: {container}/{name}'.format(container=container, name=name))
        return True
    if not (file_md5 or file_sha256):
        file_md5, file_sha256 = utils._get_file_hashes(filename)
    md5_key = metadata.get(self._connection._OBJECT_MD5_KEY, metadata.get(self._connection._SHADE_OBJECT_MD5_KEY, ''))
    sha256_key = metadata.get(self._connection._OBJECT_SHA256_KEY, metadata.get(self._connection._SHADE_OBJECT_SHA256_KEY, ''))
    up_to_date = utils._hashes_up_to_date(md5=file_md5, sha256=file_sha256, md5_key=md5_key, sha256_key=sha256_key)
    if not up_to_date:
        self._connection.log.debug('swift checksum mismatch:  %(filename)s!=%(container)s/%(name)s', {'filename': filename, 'container': container, 'name': name})
        return True
    self._connection.log.debug('swift object up to date: %(container)s/%(name)s', {'container': container, 'name': name})
    return False