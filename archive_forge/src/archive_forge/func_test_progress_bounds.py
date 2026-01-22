import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_progress_bounds():
    progress = Progress()
    progress.max = 200
    assert progress.param.value.bounds == (-1, 200)
    progress.value = 120
    assert progress.value == 120
    progress.value = -1
    assert progress.value == -1