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
def get_current_branch(cls, location: str) -> Optional[str]:
    """
        Return the current branch, or None if HEAD isn't at a branch
        (e.g. detached HEAD).
        """
    args = ['symbolic-ref', '-q', 'HEAD']
    output = cls.run_command(args, extra_ok_returncodes=(1,), show_stdout=False, stdout_only=True, cwd=location)
    ref = output.strip()
    if ref.startswith('refs/heads/'):
        return ref[len('refs/heads/'):]
    return None