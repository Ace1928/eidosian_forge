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
def del_object(self, name):
    """Delete an object.

        Args:
          name: The object name
        Raises:
          SwiftException: if unable to delete
        """
    path = self.base_path + '/' + name
    ret = self.httpclient.request('DELETE', path)
    if ret.status_code < 200 or ret.status_code > 300:
        raise SwiftException('DELETE request failed with error code %s' % ret.status_code)