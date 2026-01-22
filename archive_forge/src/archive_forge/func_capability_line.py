import collections
import os
import socket
import sys
import time
from typing import List, Tuple, Dict, Optional, Iterable
import zlib
import socketserver
from dulwich.archive import tar_stream
from dulwich.errors import (
from dulwich import log_utils
from dulwich.objects import (
from dulwich.pack import (
from dulwich.protocol import (  # noqa: F401
from dulwich.refs import (
from dulwich.repo import (
@classmethod
def capability_line(cls, capabilities):
    logger.info('Sending capabilities: %s', capabilities)
    return b''.join([b' ' + c for c in capabilities])