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
@classmethod
def init_bare(cls, scon, conf):
    """Create a new bare repository.

        Args:
          scon: a `SwiftConnector` instance
          conf: a ConfigParser object
        Returns:
          a `SwiftRepo` instance
        """
    scon.create_root()
    for obj in [posixpath.join(OBJECTDIR, PACKDIR), posixpath.join(INFODIR, 'refs')]:
        scon.put_object(obj, BytesIO(b''))
    ret = cls(scon.root, conf)
    ret._init_files(True)
    return ret