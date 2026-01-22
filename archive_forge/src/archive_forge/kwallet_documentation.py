import sys
import os
import contextlib
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError
from ..errors import PasswordSetError, InitError, KeyringLocked
from .._compat import properties
Delete the password for the username of the service.