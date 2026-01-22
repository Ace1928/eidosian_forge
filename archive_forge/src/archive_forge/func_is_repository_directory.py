import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
@classmethod
def is_repository_directory(cls, path: str) -> bool:
    """
        Return whether a directory path is a repository directory.
        """
    logger.debug('Checking in %s for %s (%s)...', path, cls.dirname, cls.name)
    return os.path.exists(os.path.join(path, cls.dirname))