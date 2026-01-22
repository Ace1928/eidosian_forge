from typing import Dict, Optional
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import SS_PREFIX
from secretstorage.dhcrypto import Session
from secretstorage.exceptions import LockedException, PromptDismissedException
from secretstorage.util import DBusAddressWrapper, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def get_secret_content_type(self) -> str:
    """Returns content type of item secret (string)."""
    self.ensure_not_locked()
    if not self.session:
        self.session = open_session(self.connection)
    secret, = self._item.call('GetSecret', 'o', self.session.object_path)
    return str(secret[3])