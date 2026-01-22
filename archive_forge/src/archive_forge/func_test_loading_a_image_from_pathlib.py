import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_loading_a_image_from_pathlib():
    """Tests the loading of a image from a pathlib"""
    filepath = Path(__file__).parent.parent / 'test_data' / 'logo.png'
    image_pane = PNG(filepath)
    image_data = image_pane._data(filepath)
    assert b'PNG' in image_data