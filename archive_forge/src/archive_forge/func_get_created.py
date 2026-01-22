from typing import Dict, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, PromptDismissedException
from secretstorage.util import DBusAddressWrapper, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def get_created(self) -> int:
    """Returns UNIX timestamp (integer) representing the time
        when the item was created.

        .. versionadded:: 1.1"""
    created = self._item.get_property('Created')
    assert isinstance(created, int)
    return created