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
def resolve_revision(cls, dest: str, url: HiddenText, rev_options: RevOptions) -> RevOptions:
    """
        Resolve a revision to a new RevOptions object with the SHA1 of the
        branch, tag, or ref if found.

        Args:
          rev_options: a RevOptions object.
        """
    rev = rev_options.arg_rev
    assert rev is not None
    sha, is_branch = cls.get_revision_sha(dest, rev)
    if sha is not None:
        rev_options = rev_options.make_new(sha)
        rev_options.branch_name = rev if is_branch else None
        return rev_options
    if not looks_like_hash(rev):
        logger.warning("Did not find branch or tag '%s', assuming revision or ref.", rev)
    if not cls._should_fetch(dest, rev):
        return rev_options
    cls.run_command(make_command('fetch', '-q', url, rev_options.to_args()), cwd=dest)
    sha = cls.get_revision(dest, rev='FETCH_HEAD')
    rev_options = rev_options.make_new(sha)
    return rev_options