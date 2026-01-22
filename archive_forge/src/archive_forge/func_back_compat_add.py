from functools import wraps
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import units
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
def back_compat_add(store_add_fun):
    """
    Provides backward compatibility for the 0.26.0+ Store.add() function.
    In 0.26.0, the 'hashing_algo' parameter is introduced and Store.add()
    returns a 5-tuple containing a computed 'multihash' value.

    This wrapper behaves as follows:

    If no hashing_algo identifier is supplied as an argument, the response
    is the pre-0.26.0 4-tuple of::

    (backend_url, bytes_written, checksum, metadata_dict)

    If a hashing_algo is supplied, the response is a 5-tuple::

    (backend_url, bytes_written, checksum, multihash, metadata_dict)

    The wrapper detects the presence of a 'hashing_algo' argument both
    by examining named arguments and positionally.
    """

    @wraps(store_add_fun)
    def add_adapter(*args, **kwargs):
        """
        Wrapper for the store 'add' function.  If no hashing_algo identifier
        is supplied, the response is the pre-0.25.0 4-tuple of::

        (backend_url, bytes_written, checksum, metadata_dict)

        If a hashing_algo is supplied, the response is a 5-tuple::

        (backend_url, bytes_written, checksum, multihash, metadata_dict)
        """
        back_compat_required = True
        p_algo = 4
        max_args = 7
        num_args = len(args)
        num_kwargs = len(kwargs)
        if num_args + num_kwargs == max_args:
            back_compat_required = False
        elif 'hashing_algo' in kwargs or (num_args >= p_algo + 1 and isinstance(args[p_algo], str)):
            back_compat_required = False
        elif kwargs and 'image_' in ''.join(kwargs):
            kwargs['hashing_algo'] = 'md5'
        else:
            args = args[:p_algo] + ('md5',) + args[p_algo:]
        backend_url, bytes_written, checksum, multihash, metadata_dict = store_add_fun(*args, **kwargs)
        if back_compat_required:
            return (backend_url, bytes_written, checksum, metadata_dict)
        return (backend_url, bytes_written, checksum, multihash, metadata_dict)
    return add_adapter