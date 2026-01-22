from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
def test_show_message():

    def make_callback():
        count = 0

        def f(message):
            nonlocal count
            count += 1
        return f
    f = make_callback()
    with utils.obj_in_module(callbacks, 'showmessage', f):
        assert f.__closure__[0].cell_contents == 0
        msg = openrlib.ffi.new('char []', b'foo')
        callbacks._showmessage(msg)
        assert f.__closure__[0].cell_contents == 1