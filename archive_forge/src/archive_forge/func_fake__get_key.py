import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from os_brick.encryptors import cryptsetup
from os_brick import exception
from os_brick.tests.encryptors import test_base
def fake__get_key(context, passphrase):
    raw = bytes(binascii.unhexlify(passphrase))
    symmetric_key = key.SymmetricKey('AES', len(raw) * 8, raw)
    return symmetric_key