from OpenSSL import crypto
from oauth2client_4_0 import _helpers
def pkcs12_key_as_pem(private_key_bytes, private_key_password):
    """Convert the contents of a PKCS#12 key to PEM using pyOpenSSL.

    Args:
        private_key_bytes: Bytes. PKCS#12 key in DER format.
        private_key_password: String. Password for PKCS#12 key.

    Returns:
        String. PEM contents of ``private_key_bytes``.
    """
    private_key_password = _helpers._to_bytes(private_key_password)
    pkcs12 = crypto.load_pkcs12(private_key_bytes, private_key_password)
    return crypto.dump_privatekey(crypto.FILETYPE_PEM, pkcs12.get_privatekey())