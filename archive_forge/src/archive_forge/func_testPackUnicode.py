import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testPackUnicode():
    test_data = ['', 'abcd', ['defgh'], 'Русский текст']
    for td in test_data:
        re = unpackb(packb(td), use_list=1, raw=False)
        assert re == td
        packer = Packer()
        data = packer.pack(td)
        re = Unpacker(BytesIO(data), raw=False, use_list=1).unpack()
        assert re == td