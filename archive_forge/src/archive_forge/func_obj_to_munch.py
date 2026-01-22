import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def obj_to_munch(obj):
    """Turn an object with attributes into a dict suitable for serializing.

    Some of the things that are returned in OpenStack are objects with
    attributes. That's awesome - except when you want to expose them as JSON
    structures. We use this as the basis of get_hostvars_from_server above so
    that we can just have a plain dict of all of the values that exist in the
    nova metadata for a server.
    """
    if obj is None:
        return None
    elif isinstance(obj, utils.Munch) or hasattr(obj, 'mock_add_spec'):
        return obj
    elif isinstance(obj, dict):
        instance = utils.Munch(obj)
    else:
        instance = utils.Munch()
    for key in dir(obj):
        try:
            value = getattr(obj, key)
        except AttributeError:
            continue
        if isinstance(value, NON_CALLABLES) and (not key.startswith('_')):
            instance[key] = value
    return instance