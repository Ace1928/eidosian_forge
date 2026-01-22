import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_and_munchify(key, data):
    """Get the value associated to key and convert it.

    The value will be converted in a Munch object or a list of Munch objects
    based on the type
    """
    result = data.get(key, []) if key else data
    if isinstance(result, list):
        return obj_list_to_munch(result)
    elif isinstance(result, dict):
        return obj_to_munch(result)
    return result