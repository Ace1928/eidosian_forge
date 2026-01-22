from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def connect_tls(*args, **kwargs):
    """Initialize a :class:`Manager` over the TLS transport."""
    device_params = _extract_device_params(kwargs)
    manager_params = _extract_manager_params(kwargs)
    nc_params = _extract_nc_params(kwargs)
    device_handler = make_device_handler(device_params)
    device_handler.add_additional_netconf_params(nc_params)
    session = transport.TLSSession(device_handler)
    session.connect(*args, **kwargs)
    return Manager(session, device_handler, **manager_params)