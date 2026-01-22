from io import BytesIO
import sys
import pytest
from srsly.msgpack import Unpacker, packb, OutOfData, ExtType
@pytest.mark.skipif("not hasattr(sys, 'getrefcount') == True", reason='sys.getrefcount() is needed to pass this test')
def test_unpacker_hook_refcnt():
    result = []

    def hook(x):
        result.append(x)
        return x
    basecnt = sys.getrefcount(hook)
    up = Unpacker(object_hook=hook, list_hook=hook)
    assert sys.getrefcount(hook) >= basecnt + 2
    up.feed(packb([{}]))
    up.feed(packb([{}]))
    assert up.unpack() == [{}]
    assert up.unpack() == [{}]
    assert result == [{}, [{}], {}, [{}]]
    del up
    assert sys.getrefcount(hook) == basecnt