from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_pinfo_getindex():

    def dummy():
        """
        MARKER
        """
    container = [dummy]
    with cleanup_user_ns(container=container):
        with AssertPrints('MARKER'):
            ip._inspect('pinfo', 'container[0]', detail_level=0)
    assert 'container' not in ip.user_ns.keys()