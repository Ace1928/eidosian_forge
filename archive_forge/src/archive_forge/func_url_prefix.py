import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
def url_prefix(mat) -> str:
    """Extract the URL prefix from a regex match.

    Args:
      mat: A regex match object.
    Returns: The URL prefix, defined as the text before the match in the
        original string. Normalized to start with one leading slash and end
        with zero.
    """
    return '/' + mat.string[:mat.start()].strip('/')