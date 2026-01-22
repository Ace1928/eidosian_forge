from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def call_and_read_response(self):
    """Send the request to the server, and read the initial response.

        This doesn't read all of the body content of the response, instead it
        returns (response_tuple, response_handler). response_tuple is the 'ok',
        or 'error' information, and 'response_handler' can be used to get the
        content stream out.
        """
    self._run_call_hooks()
    protocol_version = self.client._medium._protocol_version
    if protocol_version is None:
        return self._call_determining_protocol_version()
    else:
        return self._call(protocol_version)