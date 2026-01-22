import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_image_alt_tag():
    """Simple test for display.Image(args, alt=x,)"""
    thisurl = 'http://example.com/image.png'
    img = display.Image(url=thisurl, alt='an image')
    assert '<img src="%s" alt="an image"/>' % thisurl == img._repr_html_()
    img = display.Image(url=thisurl, unconfined=True, alt='an image')
    assert '<img src="%s" class="unconfined" alt="an image"/>' % thisurl == img._repr_html_()
    img = display.Image(url=thisurl, alt='>"& <')
    assert '<img src="%s" alt="&gt;&quot;&amp; &lt;"/>' % thisurl == img._repr_html_()
    img = display.Image(url=thisurl, metadata={'alt': 'an image'})
    assert img.alt == 'an image'
    here = os.path.dirname(__file__)
    img = display.Image(os.path.join(here, '2x2.png'), alt='an image')
    assert img.alt == 'an image'
    _, md = img._repr_png_()
    assert md['alt'] == 'an image'