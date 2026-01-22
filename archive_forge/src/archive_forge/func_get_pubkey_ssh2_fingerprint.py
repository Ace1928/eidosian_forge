import hashlib
from libcloud.utils.py3 import b, hexadigits, base64_decode_string
def get_pubkey_ssh2_fingerprint(pubkey):
    if not cryptography_available:
        raise RuntimeError('cryptography is not available')
    public_key = serialization.load_ssh_public_key(b(pubkey), backend=default_backend())
    pub_der = public_key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    return _to_md5_fingerprint(pub_der)