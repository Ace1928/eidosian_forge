from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def call_with_body_bytes_expecting_body(self, method, args, body):
    """Call a method on the remote server with body bytes."""
    if not isinstance(method, bytes):
        raise TypeError('method must be a byte string, not {!r}'.format(method))
    for arg in args:
        if not isinstance(arg, bytes):
            raise TypeError('args must be byte strings, not {!r}'.format(args))
    if not isinstance(body, bytes):
        raise TypeError('body must be byte string, not {!r}'.format(body))
    response, response_handler = self._call_and_read_response(method, args, body=body, expect_response_body=True)
    return (response, response_handler)