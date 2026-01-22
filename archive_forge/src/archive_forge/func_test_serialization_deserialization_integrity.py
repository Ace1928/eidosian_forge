import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_serialization_deserialization_integrity(self):
    from ipykernel.comm import Comm
    uploader = FileUpload()
    mock_comm = MagicMock(spec=Comm)
    mock_comm.send = MagicMock()
    mock_comm.kernel = 'does not matter'
    uploader.comm = mock_comm
    message = {'value': [FILE_UPLOAD_FRONTEND_CONTENT]}
    uploader.set_state(message)
    mock_comm.send.assert_not_called()