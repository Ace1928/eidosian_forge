from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_multiple_rendered_links():
    links = ('www.a.b', 'http://a.c', 'https://a.d', 'ftp://a.e')
    df = DataFrame(['text {} {} text {} {}'.format(*links)])
    result = df.style.format(hyperlinks='html').to_html()
    href = '<a href="{0}" target="_blank">{0}</a>'
    for link in links:
        assert href.format(link) in result
    assert href.format('text') not in result