import platform
import os
import warnings
import functools
from ...backend import KeyringBackend
from ...errors import PasswordSetError
from ...errors import PasswordDeleteError
from ...errors import KeyringLocked
from ...errors import KeyringError
from ..._compat import properties
def with_keychain(self, keychain):
    warnings.warn('macOS.Keyring.with_keychain is deprecated. Use with_properties instead.', DeprecationWarning, stacklevel=2)
    return self.with_properties(keychain=keychain)