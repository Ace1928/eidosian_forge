import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
def serve_git(transport, host=None, port=None, inet=False, timeout=None):
    backend = BzrBackend(transport)
    if host is None:
        host = 'localhost'
    if port:
        server = BzrTCPGitServer(backend, host, port)
    else:
        server = BzrTCPGitServer(backend, host)
    server.serve_forever()