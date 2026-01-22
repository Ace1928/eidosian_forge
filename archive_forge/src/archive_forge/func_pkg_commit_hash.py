import os
import platform
import pprint
import sys
import subprocess
from pathlib import Path
from IPython.core import release
from IPython.utils import _sysinfo, encoding
def pkg_commit_hash(pkg_path: str) -> tuple[str, str]:
    """Get short form of commit hash given directory `pkg_path`

    We get the commit hash from (in order of preference):

    * IPython.utils._sysinfo.commit
    * git output, if we are in a git repository

    If these fail, we return a not-found placeholder tuple

    Parameters
    ----------
    pkg_path : str
        directory containing package
        only used for getting commit from active repo

    Returns
    -------
    hash_from : str
        Where we got the hash from - description
    hash_str : str
        short form of hash
    """
    if _sysinfo.commit:
        return ('installation', _sysinfo.commit)
    proc = subprocess.Popen('git rev-parse --short HEAD'.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=pkg_path)
    repo_commit, _ = proc.communicate()
    if repo_commit:
        return ('repository', repo_commit.strip().decode('ascii'))
    return ('(none found)', '<not found>')