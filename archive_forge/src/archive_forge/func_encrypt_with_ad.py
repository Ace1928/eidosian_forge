def encrypt_with_ad(self, ad, plaintext):
    """
        EncryptWithAd(ad, plaintext):
        If k is non-empty returns ENCRYPT(k, n++, ad, plaintext). Otherwise returns plaintext.

        :param ad:
        :type ad: bytes
        :param plaintext:
        :type plaintext: bytes
        :return:
        :rtype: bytes
        """
    if self._key is None:
        return plaintext
    result = self._cipher.encrypt(self._key, self._nonce, ad, plaintext)
    self._nonce += 1
    return result