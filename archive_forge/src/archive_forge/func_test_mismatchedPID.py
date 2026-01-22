import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_mismatchedPID(self) -> None:
    """
        If the current process PID does not match the PID in the
        environment then the systemd variables in the environment were set for
        a different process (perhaps our parent) and the inherited descriptors
        are not intended for this process so L{ListenFDs.inheritedDescriptors}
        returns an empty list.
        """
    env = buildEnvironment(3, os.getpid() + 1)
    sddaemon = ListenFDs.fromEnvironment(environ=env)
    self.assertEqual([], sddaemon.inheritedDescriptors())