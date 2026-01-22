from yowsup.layers.protocol_media.mediacipher import MediaCipher
import base64
import unittest
def test_decrypt_image(self):
    media_key, media_plaintext, media_ciphertext = map(base64.b64decode, self.IMAGE)
    self.assertEqual(media_plaintext, self._cipher.decrypt_image(media_ciphertext, media_key))