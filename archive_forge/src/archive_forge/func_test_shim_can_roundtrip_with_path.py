from typing import List
from thinc.shims.shim import Shim
from ..util import make_tempdir
def test_shim_can_roundtrip_with_path():
    with make_tempdir() as path:
        shim_path = path / 'cool_shim.data'
        shim = MockShim([1, 2, 3])
        shim.to_disk(shim_path)
        copy_shim = shim.from_disk(shim_path)
    assert copy_shim.to_bytes() == shim.to_bytes()