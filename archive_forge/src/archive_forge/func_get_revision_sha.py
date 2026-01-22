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
def get_revision_sha(cls, dest: str, rev: str) -> Tuple[Optional[str], bool]:
    """
        Return (sha_or_none, is_branch), where sha_or_none is a commit hash
        if the revision names a remote branch or tag, otherwise None.

        Args:
          dest: the repository directory.
          rev: the revision name.
        """
    output = cls.run_command(['show-ref', rev], cwd=dest, show_stdout=False, stdout_only=True, on_returncode='ignore')
    refs = {}
    for line in output.strip().split('\n'):
        line = line.rstrip('\r')
        if not line:
            continue
        try:
            ref_sha, ref_name = line.split(' ', maxsplit=2)
        except ValueError:
            raise ValueError(f'unexpected show-ref line: {line!r}')
        refs[ref_name] = ref_sha
    branch_ref = f'refs/remotes/origin/{rev}'
    tag_ref = f'refs/tags/{rev}'
    sha = refs.get(branch_ref)
    if sha is not None:
        return (sha, True)
    sha = refs.get(tag_ref)
    return (sha, False)