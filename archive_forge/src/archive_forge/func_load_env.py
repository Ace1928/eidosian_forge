import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def load_env() -> typing.Optional[backend.KeyringBackend]:
    """Load a keyring configured in the environment variable."""
    try:
        return load_keyring(os.environ['PYTHON_KEYRING_BACKEND'])
    except KeyError:
        return None