import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_quoted_space_in_selector_name(self):
    html = '<div style="display: wrong">nope</div>\n        <div style="display: right">yes</div>\n        '
    soup = BeautifulSoup(html, 'html.parser')
    [chosen] = soup.select('div[style="display: right"]')
    assert 'yes' == chosen.string