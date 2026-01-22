from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def call_expecting_body(self, method, *args):
    """Call a method and return the result and the protocol object.

        The body can be read like so::

            result, smart_protocol = smart_client.call_expecting_body(...)
            body = smart_protocol.read_body_bytes()
        """
    return self._call_and_read_response(method, args, expect_response_body=True)