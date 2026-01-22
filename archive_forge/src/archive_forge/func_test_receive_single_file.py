import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_receive_single_file(self):
    uploader = FileUpload()
    message = {'value': [FILE_UPLOAD_FRONTEND_CONTENT]}
    uploader.set_state(message)
    assert len(uploader.value) == 1
    uploaded_file, = uploader.value
    assert uploaded_file.name == 'file-name.txt'
    assert uploaded_file.type == 'text/plain'
    assert uploaded_file.size == 20760
    assert uploaded_file.content.tobytes() == b'file content'
    assert uploaded_file.last_modified == dt.datetime(2020, 1, 9, 13, 58, 16, 434000, tzinfo=dt.timezone.utc)