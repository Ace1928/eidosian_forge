import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def body_required(req, resp):
    """Render Hello world or set 411."""
    if req.environ.get('Content-Length', None) is None:
        resp.status = '411 Length Required'
        return
    return 'Hello world!'