import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def curve_user_id(self, client_public_key: bytes) -> str:
    """Return the User-Id corresponding to a CURVE client's public key

        Default implementation uses the z85-encoding of the public key.

        Override to define a custom mapping of public key : user-id

        This is only called on successful authentication.

        Parameters
        ----------
        client_public_key: bytes
            The client public key used for the given message

        Returns
        -------
        user_id: unicode
            The user ID as text
        """
    return z85.encode(client_public_key).decode('ascii')