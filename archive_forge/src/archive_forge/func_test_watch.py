import asyncio
import os
import pathlib
import tempfile
from panel.io.location import Location
from panel.io.reload import (
from panel.io.state import state
from panel.tests.util import async_wait_until
def test_watch():
    filepath = os.path.abspath(__file__)
    watch(filepath)
    assert _watched_files == {filepath}
    _watched_files.clear()