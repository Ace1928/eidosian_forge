import unittest
from Cryptodome.IO._PBES import PBES2
def test6(self):
    ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA512-224AndAES192-GCM')
    pt = PBES2.decrypt(ct, self.passphrase)
    self.assertEqual(self.ref, pt)