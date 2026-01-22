import logging
import os
import re
from typing import List, Optional, Tuple
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
def call_vcs_version(self) -> Tuple[int, ...]:
    """Query the version of the currently installed Subversion client.

        :return: A tuple containing the parts of the version information or
            ``()`` if the version returned from ``svn`` could not be parsed.
        :raises: BadCommand: If ``svn`` is not installed.
        """
    version_prefix = 'svn, version '
    version = self.run_command(['--version'], show_stdout=False, stdout_only=True)
    if not version.startswith(version_prefix):
        return ()
    version = version[len(version_prefix):].split()[0]
    version_list = version.partition('-')[0].split('.')
    try:
        parsed_version = tuple(map(int, version_list))
    except ValueError:
        return ()
    return parsed_version