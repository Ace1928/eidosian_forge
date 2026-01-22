import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def buildEnvironment(count: int, pid: object) -> Dict[str, str]:
    """
    @param count: The number of file descriptors to indicate as inherited.

    @param pid: The pid of the inheriting process to indicate.

    @return: A copy of the current process environment with the I{systemd}
        file descriptor inheritance-related environment variables added to it.
    """
    result = os.environ.copy()
    result['LISTEN_FDS'] = str(count)
    result['LISTEN_FDNAMES'] = ':'.join([f'{n}.socket' for n in range(count)])
    result['LISTEN_PID'] = str(pid)
    return result