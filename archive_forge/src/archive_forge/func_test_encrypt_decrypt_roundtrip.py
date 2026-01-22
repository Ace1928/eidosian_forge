import matplotlib._type1font as t1f
import os.path
import difflib
import pytest
def test_encrypt_decrypt_roundtrip():
    data = b'this is my plaintext \x00\x01\x02\x03'
    encrypted = t1f.Type1Font._encrypt(data, 'eexec')
    decrypted = t1f.Type1Font._decrypt(encrypted, 'eexec')
    assert encrypted != decrypted
    assert data == decrypted