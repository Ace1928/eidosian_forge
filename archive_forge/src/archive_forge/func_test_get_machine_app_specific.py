import contextlib
import errno
import uuid
import pytest
from systemd import id128
def test_get_machine_app_specific():
    a1 = uuid.uuid1()
    a2 = uuid.uuid1()
    with skip_oserror(errno.ENOSYS):
        u1 = id128.get_machine_app_specific(a1)
    u2 = id128.get_machine_app_specific(a2)
    u3 = id128.get_machine_app_specific(a1)
    u4 = id128.get_machine_app_specific(a2)
    assert u1 != u2
    assert u1 == u3
    assert u2 == u4