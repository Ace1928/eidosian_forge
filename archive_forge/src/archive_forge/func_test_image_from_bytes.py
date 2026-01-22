import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_image_from_bytes():
    path = os.path.dirname(__file__)
    with open(os.path.join(path, '../test_data/logo.png'), 'rb') as f:
        img = f.read()
    image_pane = PNG(img)
    image_data = image_pane._data(img)
    assert b'PNG' in image_data