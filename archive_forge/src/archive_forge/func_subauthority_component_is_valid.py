from . import exceptions
from . import misc
from . import normalizers
def subauthority_component_is_valid(uri, component):
    """Determine if the userinfo, host, and port are valid."""
    try:
        subauthority_dict = uri.authority_info()
    except exceptions.InvalidAuthority:
        return False
    if component == 'host':
        return host_is_valid(subauthority_dict['host'])
    elif component != 'port':
        return True
    try:
        port = int(subauthority_dict['port'])
    except TypeError:
        return True
    return 0 <= port <= 65535