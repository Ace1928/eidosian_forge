import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def test_odict():
    seq = [(b'one', 1), (b'two', 2), (b'three', 3), (b'four', 4)]
    od = OrderedDict(seq)
    assert unpackb(packb(od), use_list=1) == dict(seq)

    def pair_hook(seq):
        return list(seq)
    assert unpackb(packb(od), object_pairs_hook=pair_hook, use_list=1) == seq