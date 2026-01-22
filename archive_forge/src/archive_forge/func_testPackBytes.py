import struct
import pytest
from collections import OrderedDict
from io import BytesIO
from srsly.msgpack import packb, unpackb, Unpacker, Packer
def testPackBytes():
    test_data = [b'', b'abcd', (b'defgh',)]
    for td in test_data:
        check(td)