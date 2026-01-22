import logging
import os.path
import pathlib
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.vcs.versioncontrol import (
@classmethod
def has_commit(cls, location: str, rev: str) -> bool:
    """
        Check if rev is a commit that is available in the local repository.
        """
    try:
        cls.run_command(['rev-parse', '-q', '--verify', 'sha^' + rev], cwd=location, log_failed_cmd=False)
    except InstallationError:
        return False
    else:
        return True