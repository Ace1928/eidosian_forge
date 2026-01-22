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
def get_loose_object(req, backend, mat):
    sha = (mat.group(1) + mat.group(2)).encode('ascii')
    logger.info('Sending loose object %s', sha)
    object_store = get_repo(backend, mat).object_store
    if not object_store.contains_loose(sha):
        yield req.not_found('Object not found')
        return
    try:
        data = object_store[sha].as_legacy_object()
    except OSError:
        yield req.error('Error reading object')
        return
    req.cache_forever()
    req.respond(HTTP_OK, 'application/x-git-loose-object')
    yield data