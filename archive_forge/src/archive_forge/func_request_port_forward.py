import os
import socket
import sys
import threading
import time
import weakref
from hashlib import md5, sha1, sha256, sha512
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes
import paramiko
from paramiko import util
from paramiko.auth_handler import AuthHandler, AuthOnlyHandler
from paramiko.ssh_gss import GSSAuth
from paramiko.channel import Channel
from paramiko.common import (
from paramiko.compress import ZlibCompressor, ZlibDecompressor
from paramiko.dsskey import DSSKey
from paramiko.ed25519key import Ed25519Key
from paramiko.kex_curve25519 import KexCurve25519
from paramiko.kex_gex import KexGex, KexGexSHA256
from paramiko.kex_group1 import KexGroup1
from paramiko.kex_group14 import KexGroup14, KexGroup14SHA256
from paramiko.kex_group16 import KexGroup16SHA512
from paramiko.kex_ecdh_nist import KexNistp256, KexNistp384, KexNistp521
from paramiko.kex_gss import KexGSSGex, KexGSSGroup1, KexGSSGroup14
from paramiko.message import Message
from paramiko.packet import Packetizer, NeedRekeyException
from paramiko.primes import ModulusPack
from paramiko.rsakey import RSAKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.server import ServerInterface
from paramiko.sftp_client import SFTPClient
from paramiko.ssh_exception import (
from paramiko.util import (
import atexit
def request_port_forward(self, address, port, handler=None):
    """
        Ask the server to forward TCP connections from a listening port on
        the server, across this SSH session.

        If a handler is given, that handler is called from a different thread
        whenever a forwarded connection arrives.  The handler parameters are::

            handler(
                channel,
                (origin_addr, origin_port),
                (server_addr, server_port),
            )

        where ``server_addr`` and ``server_port`` are the address and port that
        the server was listening on.

        If no handler is set, the default behavior is to send new incoming
        forwarded connections into the accept queue, to be picked up via
        `accept`.

        :param str address: the address to bind when forwarding
        :param int port:
            the port to forward, or 0 to ask the server to allocate any port
        :param callable handler:
            optional handler for incoming forwarded connections, of the form
            ``func(Channel, (str, int), (str, int))``.

        :return: the port number (`int`) allocated by the server

        :raises:
            `.SSHException` -- if the server refused the TCP forward request
        """
    if not self.active:
        raise SSHException('SSH session not active')
    port = int(port)
    response = self.global_request('tcpip-forward', (address, port), wait=True)
    if response is None:
        raise SSHException('TCP forwarding request denied')
    if port == 0:
        port = response.get_int()
    if handler is None:

        def default_handler(channel, src_addr, dest_addr_port):
            self._queue_incoming_channel(channel)
        handler = default_handler
    self._tcp_handler = handler
    return port