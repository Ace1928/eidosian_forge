import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_load_from_byteio():
    """Testing a loading a image from a ByteIo"""
    memory = BytesIO()
    path = os.path.dirname(__file__)
    with open(os.path.join(path, '../test_data/logo.png'), 'rb') as image_file:
        memory.write(image_file.read())
    image_pane = PNG(memory)
    image_data = image_pane._data(memory)
    assert b'PNG' in image_data