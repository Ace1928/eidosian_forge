from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
@classmethod
def authorization_engine_factory(cls, *args):
    return NoNetworkAuthorizationEngine(*args)