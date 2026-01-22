from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
def write_private_key(self, file_obj, password=None):
    self._write_private_key(file_obj, self.key, serialization.PrivateFormat.TraditionalOpenSSL, password=password)