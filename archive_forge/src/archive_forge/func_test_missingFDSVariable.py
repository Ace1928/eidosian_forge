import os
from typing import Dict, Mapping, Sequence
from hamcrest import assert_that, equal_to, not_
from hypothesis import given
from hypothesis.strategies import dictionaries, integers, lists
from twisted.python.systemd import ListenFDs
from twisted.trial.unittest import SynchronousTestCase
from .strategies import systemdDescriptorNames
def test_missingFDSVariable(self) -> None:
    """
        If the I{LISTEN_FDS} and I{LISTEN_FDNAMES} environment variables
        are not present, no inherited descriptors are reported.
        """
    env = buildEnvironment(3, os.getpid())
    del env['LISTEN_FDS']
    del env['LISTEN_FDNAMES']
    sddaemon = ListenFDs.fromEnvironment(environ=env)
    self.assertEqual([], sddaemon.inheritedDescriptors())