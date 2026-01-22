from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
def blob_size(self, digest):
    """Override."""
    if digest not in self._layer_to_filename:
        return self._checked_legacy_base.blob_size(digest)
    info = os.stat(self._layer_to_filename[digest])
    return info.st_size