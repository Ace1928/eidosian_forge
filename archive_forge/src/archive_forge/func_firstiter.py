import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
def firstiter(agen):
    events.append('firstiter {}'.format(agen.ag_frame.f_locals['ident']))