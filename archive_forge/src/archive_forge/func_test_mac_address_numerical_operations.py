import pickle
import random
from netaddr import (
def test_mac_address_numerical_operations():
    mac = EUI('00-1B-77-49-54-FD')
    assert int(mac) == 117965411581
    assert hex(mac) == '0x1b774954fd'
    assert mac.bits() == '00000000-00011011-01110111-01001001-01010100-11111101'
    assert mac.bin == '0b1101101110111010010010101010011111101'