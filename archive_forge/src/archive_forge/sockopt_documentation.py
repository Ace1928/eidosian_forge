import platform
import socket
import struct
from os_ken.lib import sockaddr
Enable TCP-MD5 on the given socket.

    :param s: Socket
    :param addr: Associated address.  On some platforms, this has no effect.
    :param key: Key.  On some platforms, this has no effect.
    