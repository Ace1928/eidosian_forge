import sys
from ..volumeutils import endian_codes, native_code, swapped_code
def test_native_swapped():
    native_is_le = sys.byteorder == 'little'
    if native_is_le:
        assert (native_code, swapped_code) == ('<', '>')
    else:
        assert (native_code, swapped_code) == ('>', '<')