import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_unpack_tell():
    stream = io.BytesIO()
    messages = [2 ** i - 1 for i in range(65)]
    messages += [-2 ** i for i in range(1, 64)]
    messages += [b'hello', b'hello' * 1000, list(range(20)), {i: bytes(i) * i for i in range(10)}, {i: bytes(i) * i for i in range(32)}]
    offsets = []
    for m in messages:
        pack(m, stream)
        offsets.append(stream.tell())
    stream.seek(0)
    unpacker = Unpacker(stream)
    for m, o in zip(messages, offsets):
        m2 = next(unpacker)
        assert m == m2
        assert o == unpacker.tell()