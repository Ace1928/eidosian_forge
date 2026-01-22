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
@mock.patch('warnings.warn')
def test_encourage_iframe_over_html(m_warn):
    display.HTML()
    m_warn.assert_not_called()
    display.HTML('<br />')
    m_warn.assert_not_called()
    display.HTML('<html><p>Lots of content here</p><iframe src="http://a.com"></iframe>')
    m_warn.assert_not_called()
    display.HTML('<iframe src="http://a.com"></iframe>')
    m_warn.assert_called_with('Consider using IPython.display.IFrame instead')
    m_warn.reset_mock()
    display.HTML('<IFRAME SRC="http://a.com"></IFRAME>')
    m_warn.assert_called_with('Consider using IPython.display.IFrame instead')