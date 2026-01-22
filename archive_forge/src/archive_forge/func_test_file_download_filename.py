import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_filename(tmpdir):
    file_download = FileDownload()
    filepath = tmpdir.join('foo.txt')
    filepath.write('content')
    file_download.file = str(filepath)
    assert file_download.filename == 'foo.txt'
    file_download._clicks += 1
    file_download.file = __file__
    assert file_download.filename == 'test_misc.py'
    file_download.file = StringIO('data')
    assert file_download.filename == 'test_misc.py'