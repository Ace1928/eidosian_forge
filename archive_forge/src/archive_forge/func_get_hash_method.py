import binascii
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography import x509
from oslo_log import log as logging
from oslo_serialization import base64
from oslo_utils import encodeutils
from cursive import exception
from cursive.i18n import _, _LE
from cursive import verifiers
def get_hash_method(hash_method_name):
    """Verify the hash method name and create the hash method.

    :param hash_method_name: the name of the hash method to retrieve
    :returns: the hash method, a cryptography object
    :raises: SignatureVerificationError if the hash method name is invalid
    """
    if hash_method_name not in HASH_METHODS:
        raise exception.SignatureVerificationError(reason=_('Invalid signature hash method: %s') % hash_method_name)
    return HASH_METHODS[hash_method_name]