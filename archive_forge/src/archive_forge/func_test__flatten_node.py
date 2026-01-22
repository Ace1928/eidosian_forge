import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__flatten_node(self):
    self.assertFlattened(b'key\x00\x00value\n', (b'key',), b'value', [])
    self.assertFlattened(b'key\x00tuple\x00\x00value str\n', (b'key', b'tuple'), b'value str', [])
    self.assertFlattened(b'key\x00tuple\x00triple\x00\x00value str\n', (b'key', b'tuple', b'triple'), b'value str', [])
    self.assertFlattened(b'k\x00t\x00s\x00ref\x00value str\n', (b'k', b't', b's'), b'value str', [[(b'ref',)]])
    self.assertFlattened(b'key\x00tuple\x00ref\x00key\x00value str\n', (b'key', b'tuple'), b'value str', [[(b'ref', b'key')]])
    self.assertFlattened(b'00\x0000\x00\t00\x00ref00\x00value:0\n', (b'00', b'00'), b'value:0', ((), ((b'00', b'ref00'),)))
    self.assertFlattened(b'00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n', (b'00', b'11'), b'value:1', (((b'00', b'ref00'),), ((b'00', b'ref00'), (b'01', b'ref01'))))
    self.assertFlattened(b'11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n', (b'11', b'33'), b'value:3', (((b'11', b'ref22'),), ((b'11', b'ref22'), (b'11', b'ref22'))))
    self.assertFlattened(b'11\x0044\x00\t11\x00ref00\x00value:4\n', (b'11', b'44'), b'value:4', ((), ((b'11', b'ref00'),)))