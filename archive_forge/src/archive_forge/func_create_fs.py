import re
from threading import RLock
from typing import Any, Dict, Tuple
from urllib.parse import urlparse
from triad.utils.hash import to_uuid
import fs
from fs import memoryfs, open_fs, tempfs
from fs.base import FS as FSBase
from fs.glob import BoundGlobber, Globber
from fs.mountfs import MountFS
from fs.subfs import SubFS
def create_fs(self, root: str) -> FSBase:
    """create a PyFileSystem instance from `root`. `root` is in the
        format of `/` if local path, else `<scheme>://<netloc>`.
        You should override this method to provide custom instances, for
        example, if you want to create an S3FS with certain parameters.
        :param root: `/` if local path, else `<scheme>://<netloc>`
        """
    if root.startswith('temp://'):
        fs = tempfs.TempFS(root[len('temp://'):])
        return fs
    if root.startswith('mem://'):
        fs = memoryfs.MemoryFS()
        return fs
    return open_fs(root)