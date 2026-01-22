from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
The WSGI HPSS glue allows access to the whole WSGI backing
        transport, regardless of which HTTP path the request was delivered
        to.
        