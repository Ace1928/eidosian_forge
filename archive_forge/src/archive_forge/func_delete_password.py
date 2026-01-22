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
@warn_keychain
def delete_password(self, service, username):
    if username is None:
        username = ''
    try:
        return api.delete_generic_password(self.keychain, service, username)
    except api.Error as e:
        raise PasswordDeleteError("Can't delete password in keychain: {}".format(e))