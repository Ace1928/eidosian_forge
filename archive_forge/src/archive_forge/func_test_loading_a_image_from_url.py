import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
def test_loading_a_image_from_url():
    """Tests the loading of a image from a url"""
    url = 'https://raw.githubusercontent.com/holoviz/panel/main/doc/_static/logo.png'
    image_pane = PNG(url)
    image_data = image_pane._data(url)
    assert b'PNG' in image_data