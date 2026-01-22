import os
from base64 import b64decode, b64encode
from io import BytesIO, StringIO
from pathlib import Path
import pytest
from requests.exceptions import MissingSchema
from panel.pane import (
from panel.pane.markup import escape
@pytest.mark.parametrize('t', [PNG, JPG, GIF, ICO, WebP], ids=lambda t: t.name.lower())
def test_imgshape(t):
    w, h = t._imgshape(b64decode(twopixel[t.name.lower()]))
    assert w == 2
    assert h == 1