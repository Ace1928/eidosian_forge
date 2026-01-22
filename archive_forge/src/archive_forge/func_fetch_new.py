import configparser
import logging
import os
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
def fetch_new(self, dest: str, url: HiddenText, rev_options: RevOptions, verbosity: int) -> None:
    rev_display = rev_options.to_display()
    logger.info('Cloning hg %s%s to %s', url, rev_display, display_path(dest))
    if verbosity <= 0:
        flags: Tuple[str, ...] = ('--quiet',)
    elif verbosity == 1:
        flags = ()
    elif verbosity == 2:
        flags = ('--verbose',)
    else:
        flags = ('--verbose', '--debug')
    self.run_command(make_command('clone', '--noupdate', *flags, url, dest))
    self.run_command(make_command('update', *flags, rev_options.to_args()), cwd=dest)