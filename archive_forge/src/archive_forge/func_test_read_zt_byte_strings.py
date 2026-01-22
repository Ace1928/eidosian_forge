import pytest
from ..fileutils import read_zt_byte_strings
from ..tmpdirs import InTemporaryDirectory
def test_read_zt_byte_strings():
    binary = b'test.fmr\x00test.prt\x00something'
    with InTemporaryDirectory():
        path = 'test.bin'
        fwrite = open(path, 'wb')
        fwrite.write(binary)
        fwrite.close()
        fread = open(path, 'rb')
        assert read_zt_byte_strings(fread) == [b'test.fmr']
        assert fread.tell() == 9
        fread.seek(0)
        assert read_zt_byte_strings(fread, 2) == [b'test.fmr', b'test.prt']
        assert fread.tell() == 18
        fread.seek(0)
        with pytest.raises(ValueError):
            read_zt_byte_strings(fread, 3)
        fread.seek(9)
        with pytest.raises(ValueError):
            read_zt_byte_strings(fread, 2)
        fread.seek(0)
        assert read_zt_byte_strings(fread, 2, 4) == [b'test.fmr', b'test.prt']
        fread.close()