import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_callback():
    with pytest.raises(ValueError):
        FileDownload(callback=lambda: StringIO('data'))
    file_download = FileDownload(callback=lambda: StringIO('data'), filename='abc.py')
    assert file_download.data is None
    file_download._clicks += 1
    assert file_download.data is not None
    file_download.data = None

    def cb():
        file_download.filename = 'cba.py'
        return StringIO('data')
    file_download.callback = cb
    file_download._clicks += 1
    assert file_download.data is not None
    assert file_download.filename == 'cba.py'
    assert file_download.label == 'Download cba.py'