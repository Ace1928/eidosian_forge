import unittest
import websocket as ws
from websocket._abnf import *
def testMask(self):
    abnf_none_data = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_PING, mask_value=1, data=None)
    bytes_val = b'aaaa'
    self.assertEqual(abnf_none_data._get_masked(bytes_val), bytes_val)
    abnf_str_data = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_PING, mask_value=1, data='a')
    self.assertEqual(abnf_str_data._get_masked(bytes_val), b'aaaa\x00')