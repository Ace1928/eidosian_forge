from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
def create_jws_keypair(private_key_path, public_key_path):
    """Create an ECDSA key pair using an secp256r1, or NIST P-256, curve.

    :param private_key_path: location to save the private key
    :param public_key_path: location to save the public key

    """
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    with open(private_key_path, 'wb') as f:
        f.write(private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()))
    public_key = private_key.public_key()
    with open(public_key_path, 'wb') as f:
        f.write(public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo))