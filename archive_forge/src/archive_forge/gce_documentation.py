import logging
import warnings
from six.moves import http_client
from oauth2client import client
from oauth2client.contrib import _metadata
Cryptographically sign a blob (of bytes).

        This method is provided to support a common interface, but
        the actual key used for a Google Compute Engine service account
        is not available, so it can't be used to sign content.

        Args:
            blob: bytes, Message to be signed.

        Raises:
            NotImplementedError, always.
        