from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import string
import subprocess
import sys
import tarfile
import tempfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
import httplib2
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
def raw_tags(self):
    """Dictionary of tag to image id."""
    return self._tags