import os
import sys
import pytest
from .. import (
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='curio broken on 3.12 (https://github.com/python-trio/sniffio/pull/42)')
def test_curio():
    import curio
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
    ran = []

    async def this_is_curio():
        assert current_async_library() == 'curio'
        assert current_async_library() == 'curio'
        ran.append(True)
    curio.run(this_is_curio)
    assert ran == [True]
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()