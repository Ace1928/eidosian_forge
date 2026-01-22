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
def make_wsgi_chain(*args, **kwargs):
    """Factory function to create an instance of HTTPGitApplication,
    correctly wrapped with needed middleware.
    """
    app = HTTPGitApplication(*args, **kwargs)
    wrapped_app = LimitedInputFilter(GunzipFilter(app))
    return wrapped_app