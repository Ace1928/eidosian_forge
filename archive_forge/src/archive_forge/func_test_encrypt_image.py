from yowsup.layers.protocol_media.mediacipher import MediaCipher
import base64
import unittest
def test_encrypt_image(self):
    media_key, media_plaintext, media_ciphertext = map(base64.b64decode, self.IMAGE)
    encrypted = self._cipher.encrypt(media_plaintext, media_key, MediaCipher.INFO_IMAGE)
    self.assertEqual(media_ciphertext, encrypted)