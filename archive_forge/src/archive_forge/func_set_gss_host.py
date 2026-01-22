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
def set_gss_host(self, gss_host, trust_dns=True, gssapi_requested=True):
    """
        Normalize/canonicalize ``self.gss_host`` depending on various factors.

        :param str gss_host:
            The explicitly requested GSS-oriented hostname to connect to (i.e.
            what the host's name is in the Kerberos database.) Defaults to
            ``self.hostname`` (which will be the 'real' target hostname and/or
            host portion of given socket object.)
        :param bool trust_dns:
            Indicates whether or not DNS is trusted; if true, DNS will be used
            to canonicalize the GSS hostname (which again will either be
            ``gss_host`` or the transport's default hostname.)
            (Defaults to True due to backwards compatibility.)
        :param bool gssapi_requested:
            Whether GSSAPI key exchange or authentication was even requested.
            If not, this is a no-op and nothing happens
            (and ``self.gss_host`` is not set.)
            (Defaults to True due to backwards compatibility.)
        :returns: ``None``.
        """
    if not gssapi_requested:
        return
    if gss_host is None:
        gss_host = self.hostname
    if trust_dns and gss_host is not None:
        gss_host = socket.getfqdn(gss_host)
    self.gss_host = gss_host