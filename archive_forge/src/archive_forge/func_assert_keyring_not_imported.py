from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
def assert_keyring_not_imported():
    assert getattr(launchpadlib.credentials, 'keyring', missing) is missing, 'During tests the real keyring module should never be imported.'