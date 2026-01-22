import os
from glance.common import crypt
from glance.common import utils
from glance.tests import utils as test_utils
def test_encryption(self):
    key_list = ['1234567890abcdef', '12345678901234567890abcd', '1234567890abcdef1234567890ABCDEF']
    plaintext_list = ['']
    blocksize = 64
    for i in range(3 * blocksize):
        text = os.urandom(i).decode('latin1')
        plaintext_list.append(text)
    for key in key_list:
        for plaintext in plaintext_list:
            ciphertext = crypt.urlsafe_encrypt(key, plaintext, blocksize)
            self.assertIsInstance(ciphertext, str)
            self.assertNotEqual(ciphertext, plaintext)
            text = crypt.urlsafe_decrypt(key, ciphertext)
            self.assertIsInstance(text, str)
            self.assertEqual(plaintext, text)