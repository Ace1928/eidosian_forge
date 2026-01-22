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
def distributable_blob_set(self):
    """The unique set of blobs which are distributable."""
    manifest = json.loads(self.manifest())
    distributable_blobs = {x['digest'] for x in reversed(manifest['layers']) if x['mediaType'] not in docker_http.NON_DISTRIBUTABLE_LAYER_MIMES}
    distributable_blobs.add(self.config_blob())
    return distributable_blobs