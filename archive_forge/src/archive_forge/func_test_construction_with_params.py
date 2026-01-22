import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_construction_with_params(self):
    uploader = FileUpload(accept='.txt', multiple=True, disabled=True)
    assert uploader.accept == '.txt'
    assert uploader.multiple
    assert uploader.disabled