import unittest
from ...ratchet.chainkey import ChainKey
from ...kdf.hkdf import HKDF
def test_chainKeyDerivationV2(self):
    seed = bytearray([138, 183, 45, 111, 76, 197, 172, 13, 56, 126, 175, 70, 51, 120, 221, 178, 142, 221, 7, 56, 91, 28, 176, 18, 80, 199, 21, 152, 46, 122, 212, 143])
    messageKey = bytearray([2, 169, 170, 108, 125, 189, 100, 249, 211, 170, 146, 249, 42, 39, 123, 245, 70, 9, 218, 223, 11, 0, 130, 138, 207, 198, 30, 60, 114, 75, 132, 167])
    macKey = bytearray([191, 190, 94, 251, 96, 48, 48, 82, 103, 66, 227, 238, 137, 199, 2, 78, 136, 78, 68, 15, 31, 243, 118, 187, 35, 23, 178, 214, 77, 235, 124, 131])
    nextChainKey = bytearray([40, 232, 248, 254, 229, 75, 128, 30, 239, 124, 92, 251, 47, 23, 243, 44, 123, 51, 68, 133, 187, 183, 15, 172, 110, 193, 3, 66, 162, 70, 209, 93])
    chainKey = ChainKey(HKDF.createFor(2), seed, 0)
    self.assertEqual(chainKey.getKey(), seed)
    self.assertEqual(chainKey.getMessageKeys().getCipherKey(), messageKey)
    self.assertEqual(chainKey.getMessageKeys().getMacKey(), macKey)
    self.assertEqual(chainKey.getNextChainKey().getKey(), nextChainKey)
    self.assertEqual(chainKey.getIndex(), 0)
    self.assertEqual(chainKey.getMessageKeys().getCounter(), 0)
    self.assertEqual(chainKey.getNextChainKey().getIndex(), 1)
    self.assertEqual(chainKey.getNextChainKey().getMessageKeys().getCounter(), 1)