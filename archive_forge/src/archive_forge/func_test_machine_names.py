from __future__ import print_function
import select
import contextlib
import errno
from systemd import login
import pytest
def test_machine_names():
    with skip_oserror(errno.ENOENT):
        machine_names = login.machine_names()
        assert len(machine_names) >= 0