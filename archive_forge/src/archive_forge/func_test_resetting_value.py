import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_resetting_value(self):
    uploader = FileUpload()
    message = {'value': [FILE_UPLOAD_FRONTEND_CONTENT]}
    uploader.set_state(message)
    uploader.value = []
    assert uploader.get_state(key='value') == {'value': []}