import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
def get_object(self, name, range=None):
    """Retrieve an object.

        Args:
          name: The object name
          range: A string range like "0-10" to
                 retrieve specified bytes in object content
        Returns:
          A file like instance or bytestring if range is specified
        """
    headers = {}
    if range:
        headers['Range'] = 'bytes=%s' % range
    path = self.base_path + '/' + name
    ret = self.httpclient.request('GET', path, headers=headers)
    if ret.status_code == 404:
        return None
    if ret.status_code < 200 or ret.status_code > 300:
        raise SwiftException('GET request failed with error code %s' % ret.status_code)
    content = ret.read()
    if range:
        return content
    return BytesIO(content)