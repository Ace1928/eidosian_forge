from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def pick_microversion(session, required):
    """Get a new microversion if it is higher than session's default.

    :param session: The session to use for making this request.
    :type session: :class:`~keystoneauth1.adapter.Adapter`
    :param required: Minimum version that is required for an action.
    :type required: String or tuple or None.
    :return: ``required`` as a string if the ``session``'s default is too low,
        otherwise the ``session``'s default. Returns ``None`` if both
        are ``None``.
    :raises: TypeError if ``required`` is invalid.
    :raises: :class:`~openstack.exceptions.SDKException` if requested
        microversion is not supported.
    """
    if required is not None:
        required = discover.normalize_version_number(required)
    if session.default_microversion is not None:
        default = discover.normalize_version_number(session.default_microversion)
        if required is None:
            required = default
        else:
            required = default if discover.version_match(required, default) else required
    if required is not None:
        if not supports_microversion(session, required):
            raise exceptions.SDKException('Requested microversion is not supported by the server side or the default microversion is too low')
        return discover.version_to_string(required)