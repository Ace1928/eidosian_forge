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
def get_netloc_and_auth(cls, netloc: str, scheme: str) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
    """
        Parse the repository URL's netloc, and return the new netloc to use
        along with auth information.

        Args:
          netloc: the original repository URL netloc.
          scheme: the repository URL's scheme without the vcs prefix.

        This is mainly for the Subversion class to override, so that auth
        information can be provided via the --username and --password options
        instead of through the URL.  For other subclasses like Git without
        such an option, auth information must stay in the URL.

        Returns: (netloc, (username, password)).
        """
    return (netloc, (None, None))