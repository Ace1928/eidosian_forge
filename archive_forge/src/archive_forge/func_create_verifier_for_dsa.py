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
def create_verifier_for_dsa(signature, hash_method, public_key):
    """Create the verifier to use when the key type is DSA

    :param signature: the decoded signature to use
    :param hash_method: the hash method to use, as a cryptography object
    :param public_key: the public key to use, as a cryptography object
    :returns: the verifier to use to verify the signature for DSA
    """
    if not signature or not hash_method or (not public_key):
        return None
    return verifiers.DSAVerifier(signature, hash_method, public_key)