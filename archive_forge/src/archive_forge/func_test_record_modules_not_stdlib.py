import asyncio
import os
import pathlib
import tempfile
from panel.io.location import Location
from panel.io.reload import (
from panel.io.state import state
from panel.tests.util import async_wait_until
def test_record_modules_not_stdlib():
    with record_modules():
        import audioop
    assert _modules == set() or _modules == set(['audioop'])
    _modules.clear()