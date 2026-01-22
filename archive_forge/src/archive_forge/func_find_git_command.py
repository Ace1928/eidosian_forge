import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def find_git_command() -> List[str]:
    """Find command to run for system Git (usually C Git)."""
    if sys.platform == 'win32':
        try:
            import pywintypes
            import win32api
        except ImportError:
            return ['cmd', '/c', 'git']
        else:
            try:
                status, git = win32api.FindExecutable('git')
                return [git]
            except pywintypes.error:
                return ['cmd', '/c', 'git']
    else:
        return ['git']