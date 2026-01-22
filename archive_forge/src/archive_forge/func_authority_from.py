from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
def authority_from(reference, strict):
    try:
        subauthority = reference.authority_info()
    except exceptions.InvalidAuthority:
        if strict:
            raise
        userinfo, host, port = split_authority(reference.authority)
    else:
        userinfo, host, port = (subauthority.get(p) for p in ('userinfo', 'host', 'port'))
    if port:
        try:
            port = int(port)
        except ValueError:
            raise exceptions.InvalidPort(port)
    return (userinfo, host, port)