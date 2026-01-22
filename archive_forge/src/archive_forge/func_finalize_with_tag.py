from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives import ciphers
from cryptography.hazmat.primitives.ciphers import algorithms, modes
def finalize_with_tag(self, tag: bytes) -> bytes:
    tag_len = len(tag)
    if tag_len < self._mode._min_tag_length:
        raise ValueError('Authentication tag must be {} bytes or longer.'.format(self._mode._min_tag_length))
    elif tag_len > self._block_size_bytes:
        raise ValueError('Authentication tag cannot be more than {} bytes.'.format(self._block_size_bytes))
    res = self._backend._lib.EVP_CIPHER_CTX_ctrl(self._ctx, self._backend._lib.EVP_CTRL_AEAD_SET_TAG, len(tag), tag)
    self._backend.openssl_assert(res != 0)
    self._tag = tag
    return self.finalize()