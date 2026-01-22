import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def get_unique_resource_id(max_length=None):
    """
    Obtains a unique id that can be included in a resource name. This unique id is a valid
    DNS subname.

    Args:
        max_length: The maximum length of the identifier.

    Returns:
        A unique identifier that can be appended to a user-readable resource name to avoid
        naming collisions.
    """
    if max_length is not None and max_length <= 0:
        raise ValueError('The specified maximum length for the unique resource id must be positive!')
    uuid_bytes = uuid.uuid4().bytes
    uuid_b64 = base64.b64encode(uuid_bytes)
    uuid_b64 = uuid_b64.decode('ascii')
    unique_id = uuid_b64.rstrip('=\n').replace('/', '-').replace('+', 'AB').lower()
    if max_length is not None:
        unique_id = unique_id[:int(max_length)]
    return unique_id