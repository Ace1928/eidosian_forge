from base64 import b64encode, b64decode
import binascii
def base64(s: str) -> Base64String:
    """Encode the string s using Base64."""
    b: bytes = s.encode('utf-8') if isinstance(s, str) else s
    return b64encode(b).decode('ascii')