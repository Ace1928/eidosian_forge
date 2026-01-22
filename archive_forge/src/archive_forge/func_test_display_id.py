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
def test_display_id():
    ip = get_ipython()
    with mock.patch.object(ip.display_pub, 'publish') as pub:
        handle = display.display('x')
        assert handle is None
        handle = display.display('y', display_id='secret')
        assert isinstance(handle, display.DisplayHandle)
        handle2 = display.display('z', display_id=True)
        assert isinstance(handle2, display.DisplayHandle)
    assert handle.display_id != handle2.display_id
    assert pub.call_count == 3
    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('x')}, 'metadata': {}}
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('y')}, 'metadata': {}, 'transient': {'display_id': handle.display_id}}
    args, kwargs = pub.call_args_list[2]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('z')}, 'metadata': {}, 'transient': {'display_id': handle2.display_id}}