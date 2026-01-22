import binascii
import hashlib
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from paramiko.message import Message
from paramiko.common import byte_chr
from paramiko.ssh_exception import SSHException
def start_kex(self):
    self.key = X25519PrivateKey.generate()
    if self.transport.server_mode:
        self.transport._expect_packet(_MSG_KEXECDH_INIT)
        return
    m = Message()
    m.add_byte(c_MSG_KEXECDH_INIT)
    m.add_string(self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    self.transport._send_message(m)
    self.transport._expect_packet(_MSG_KEXECDH_REPLY)