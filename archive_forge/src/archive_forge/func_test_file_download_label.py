import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_label():
    file_download = FileDownload()
    assert file_download.label == 'No file set'
    file_download = FileDownload(StringIO('data'), filename='abc.py')
    assert file_download.label == 'Download abc.py'
    file_download = FileDownload(__file__)
    assert file_download.label == 'Download test_misc.py'
    file_download.auto = False
    assert file_download.label == 'Transfer test_misc.py'
    file_download.embed = True
    assert file_download.label == 'Download test_misc.py'
    file_download.filename = 'abc.py'
    assert file_download.label == 'Download abc.py'