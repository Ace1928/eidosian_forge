from binascii import hexlify
import hashlib
import logging; log = logging.getLogger(__name__)
import struct
import warnings
from passlib import exc
from passlib.utils import getrandbytes
from passlib.utils.compat import PYPY, u, bascii_to_str
from passlib.utils.decor import classproperty
from passlib.tests.utils import TestCase, skipUnless, TEST_MODE, hb
from passlib.crypto import scrypt as scrypt_mod
def test_bmix(self):
    """bmix()"""
    from passlib.crypto.scrypt._builtin import ScryptEngine
    rng = self.getRandom()

    def check_bmix(r, input, output):
        """helper to check bmix() output against reference"""
        engine = ScryptEngine(r=r, n=1 << rng.randint(1, 32), p=rng.randint(1, 1023))
        target = [rng.randint(0, 1 << 32) for _ in range(2 * r * 16)]
        engine.bmix(input, target)
        self.assertEqual(target, list(output))
        if r == 1:
            del engine.bmix
            target = [rng.randint(0, 1 << 32) for _ in range(2 * r * 16)]
            engine.bmix(input, target)
            self.assertEqual(target, list(output))
    input = unpack_uint32_list(hb('\n                f7 ce 0b 65 3d 2d 72 a4 10 8c f5 ab e9 12 ff dd\n                77 76 16 db bb 27 a7 0e 82 04 f3 ae 2d 0f 6f ad\n                89 f6 8f 48 11 d1 e8 7b cc 3b d7 40 0a 9f fd 29\n                09 4f 01 84 63 95 74 f3 9a e5 a1 31 52 17 bc d7\n\n                89 49 91 44 72 13 bb 22 6c 25 b5 4d a8 63 70 fb\n                cd 98 43 80 37 46 66 bb 8f fc b5 bf 40 c2 54 b0\n                67 d2 7c 51 ce 4a d5 fe d8 29 c9 0b 50 5a 57 1b\n                7f 4d 1c ad 6a 52 3c da 77 0e 67 bc ea af 7e 89\n            '), 32)
    output = unpack_uint32_list(hb('\n                a4 1f 85 9c 66 08 cc 99 3b 81 ca cb 02 0c ef 05\n                04 4b 21 81 a2 fd 33 7d fd 7b 1c 63 96 68 2f 29\n                b4 39 31 68 e3 c9 e6 bc fe 6b c5 b7 a0 6d 96 ba\n                e4 24 cc 10 2c 91 74 5c 24 ad 67 3d c7 61 8f 81\n\n                20 ed c9 75 32 38 81 a8 05 40 f6 4c 16 2d cd 3c\n                21 07 7c fe 5f 8d 5f e2 b1 a4 16 8f 95 36 78 b7\n                7d 3b 3d 80 3b 60 e4 ab 92 09 96 e5 9b 4d 53 b6\n                5d 2a 22 58 77 d5 ed f5 84 2c b9 f1 4e ef e4 25\n            '), 32)
    r = 2
    input = unpack_uint32_list(seed_bytes('bmix with r=2', 128 * r))
    output = unpack_uint32_list(hb('\n            ba240854954f4585f3d0573321f10beee96f12acdc1feb498131e40512934fd7\n            43e8139c17d0743c89d09ac8c3582c273c60ab85db63e410d049a9e17a42c6a1\n\n            6c7831b11bf370266afdaff997ae1286920dea1dedf0f4a1795ba710ba9017f1\n            a374400766f13ebd8969362de2d153965e9941bdde0768fa5b53e8522f116ce0\n\n            d14774afb88f46cd919cba4bc64af7fca0ecb8732d1fc2191e0d7d1b6475cb2e\n            e3db789ee478d056c4eb6c6e28b99043602dbb8dfb60c6e048bf90719da8d57d\n\n            3c42250e40ab79a1ada6aae9299b9790f767f54f388d024a1465b30cbbe9eb89\n            002d4f5c215c4259fac4d083bac5fb0b47463747d568f40bb7fa87c42f0a1dc1\n            '), 32 * r)
    check_bmix(r, input, output)
    r = 3
    input = unpack_uint32_list(seed_bytes('bmix with r=3', 128 * r))
    output = unpack_uint32_list(hb('\n            11ddd8cf60c61f59a6e5b128239bdc77b464101312c88bd1ccf6be6e75461b29\n            7370d4770c904d0b09c402573cf409bf2db47b91ba87d5a3de469df8fb7a003c\n\n            95a66af96dbdd88beddc8df51a2f72a6f588d67e7926e9c2b676c875da13161e\n            b6262adac39e6b3003e9a6fbc8c1a6ecf1e227c03bc0af3e5f8736c339b14f84\n\n            c7ae5b89f5e16d0faf8983551165f4bb712d97e4f81426e6b78eb63892d3ff54\n            80bf406c98e479496d0f76d23d728e67d2a3d2cdbc4a932be6db36dc37c60209\n\n            a5ca76ca2d2979f995f73fe8182eefa1ce0ba0d4fc27d5b827cb8e67edd6552f\n            00a5b3ab6b371bd985a158e728011314eb77f32ade619b3162d7b5078a19886c\n\n            06f12bc8ae8afa46489e5b0239954d5216967c928982984101e4a88bae1f60ae\n            3f8a456e169a8a1c7450e7955b8a13a202382ae19d41ce8ef8b6a15eeef569a7\n\n            20f54c48e44cb5543dda032c1a50d5ddf2919030624978704eb8db0290052a1f\n            5d88989b0ef931b6befcc09e9d5162320e71e80b89862de7e2f0b6c67229b93f\n            '), 32 * r)
    check_bmix(r, input, output)
    r = 4
    input = unpack_uint32_list(seed_bytes('bmix with r=4', 128 * r))
    output = unpack_uint32_list(hb('\n            803fcf7362702f30ef43250f20bc6b1b8925bf5c4a0f5a14bbfd90edce545997\n            3047bd81655f72588ca93f5c2f4128adaea805e0705a35e14417101fdb1c498c\n\n            33bec6f4e5950d66098da8469f3fe633f9a17617c0ea21275185697c0e4608f7\n            e6b38b7ec71704a810424637e2c296ca30d9cbf8172a71a266e0393deccf98eb\n\n            abc430d5f144eb0805308c38522f2973b7b6a48498851e4c762874497da76b88\n            b769b471fbfc144c0e8e859b2b3f5a11f51604d268c8fd28db55dff79832741a\n\n            1ac0dfdaff10f0ada0d93d3b1f13062e4107c640c51df05f4110bdda15f51b53\n            3a75bfe56489a6d8463440c78fb8c0794135e38591bdc5fa6cec96a124178a4a\n\n            d1a976e985bfe13d2b4af51bd0fc36dd4cfc3af08efe033b2323a235205dc43d\n            e57778a492153f9527338b3f6f5493a03d8015cd69737ee5096ad4cbe660b10f\n\n            b75b1595ddc96e3748f5c9f61fba1ef1f0c51b6ceef8bbfcc34b46088652e6f7\n            edab61521cbad6e69b77be30c9c97ea04a4af359dafc205c7878cc9a6c5d122f\n\n            8d77f3cbe65ab14c3c491ef94ecb3f5d2c2dd13027ea4c3606262bb3c9ce46e7\n            dc424729dc75f6e8f06096c0ad8ad4d549c42f0cad9b33cb95d10fb3cadba27c\n\n            5f4bf0c1ac677c23ba23b64f56afc3546e62d96f96b58d7afc5029f8168cbab4\n            533fd29fc83c8d2a32b81923992e4938281334e0c3694f0ee56f8ff7df7dc4ae\n            '), 32 * r)
    check_bmix(r, input, output)