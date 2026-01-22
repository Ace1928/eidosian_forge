import logging
from io import IOBase
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from botocore import parsers
from botocore.compat import set_socket_timeout
from botocore.exceptions import (
from botocore import ScalarTypes  # noqa
from botocore.compat import XMLParseError  # noqa
from botocore.hooks import first_non_none_response  # noqa
def set_socket_timeout(self, timeout):
    """Set the timeout seconds on the socket."""
    try:
        set_socket_timeout(self._raw_stream, timeout)
    except AttributeError:
        logger.error("Cannot access the socket object of a streaming response.  It's possible the interface has changed.", exc_info=True)
        raise