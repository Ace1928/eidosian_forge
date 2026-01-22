from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
@contextmanager
def fake_keyring(fake):
    """A context manager which injects a testing keyring implementation."""
    assert_keyring_not_imported()
    launchpadlib.credentials.keyring = fake
    launchpadlib.credentials.NoKeyringError = RuntimeError
    try:
        yield
    finally:
        del launchpadlib.credentials.keyring
        del launchpadlib.credentials.NoKeyringError