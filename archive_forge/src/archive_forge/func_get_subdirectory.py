import configparser
import logging
import os
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
@classmethod
def get_subdirectory(cls, location: str) -> Optional[str]:
    """
        Return the path to Python project root, relative to the repo root.
        Return None if the project root is in the repo root.
        """
    repo_root = cls.run_command(['root'], show_stdout=False, stdout_only=True, cwd=location).strip()
    if not os.path.isabs(repo_root):
        repo_root = os.path.abspath(os.path.join(location, repo_root))
    return find_path_to_project_root_from_repo_root(location, repo_root)