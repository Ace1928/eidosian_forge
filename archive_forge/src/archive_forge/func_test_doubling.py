import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_doubling(self):
    pointRx = 3975062381064225480574098685521977536038322097599329938779879404933190756769282040349324142327608943288400218065083035492863386391810636410549184121094237126
    pointRy = 5475657578453756240260133840068900684884763004547062714625687949872864918166460799367981360733022202781327245597758551064385199362201914511669512056775423299
    pointR = self.pointS.copy()
    pointR.double()
    self.assertEqual(pointR.x, pointRx)
    self.assertEqual(pointR.y, pointRy)
    pai = self.pointS.point_at_infinity()
    pointR = pai.copy()
    pointR.double()
    self.assertEqual(pointR, pai)
    pointR = self.pointS.copy()
    pointR += pointR
    self.assertEqual(pointR.x, pointRx)
    self.assertEqual(pointR.y, pointRy)