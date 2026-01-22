from typing import Dict, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, PromptDismissedException
from secretstorage.util import DBusAddressWrapper, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def ensure_not_locked(self) -> None:
    """If collection is locked, raises
        :exc:`~secretstorage.exceptions.LockedException`."""
    if self.is_locked():
        raise LockedException('Item is locked!')