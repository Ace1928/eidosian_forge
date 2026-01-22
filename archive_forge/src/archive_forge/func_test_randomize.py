import contextlib
import errno
import uuid
import pytest
from systemd import id128
def test_randomize():
    u1 = id128.randomize()
    u2 = id128.randomize()
    assert u1 != u2