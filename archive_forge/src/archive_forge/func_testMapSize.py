import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testMapSize(sizes=[0, 5, 50, 1000]):
    bio = BytesIO()
    packer = Packer()
    for size in sizes:
        bio.write(packer.pack_map_header(size))
        for i in range(size):
            bio.write(packer.pack(i))
            bio.write(packer.pack(i * 2))
    bio.seek(0)
    unpacker = Unpacker(bio)
    for size in sizes:
        assert unpacker.unpack() == dict(((i, i * 2) for i in range(size)))