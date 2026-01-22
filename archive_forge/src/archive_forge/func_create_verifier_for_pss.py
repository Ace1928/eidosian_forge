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
def create_verifier_for_pss(signature, hash_method, public_key):
    """Create the verifier to use when the key type is RSA-PSS.

    :param signature: the decoded signature to use
    :param hash_method: the hash method to use, as a cryptography object
    :param public_key: the public key to use, as a cryptography object
    :raises: SignatureVerificationError if the RSA-PSS specific properties
                                        are invalid
    :returns: the verifier to use to verify the signature for RSA-PSS
    """
    if not signature or not hash_method or (not public_key):
        return None
    mgf = padding.MGF1(hash_method)
    salt_length = padding.PSS.MAX_LENGTH
    return verifiers.RSAVerifier(signature, hash_method, public_key, padding.PSS(mgf=mgf, salt_length=salt_length))