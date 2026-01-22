import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
def download_action(path, known_hash):
    """
    Determine the action that is needed to get the file on disk.

    Parameters
    ----------
    path : PathLike
        The path to the file on disk.
    known_hash : str
        A known hash (checksum) of the file. Will be used to verify the
        download or check if an existing file needs to be updated. By default,
        will assume it's a SHA256 hash. To specify a different hashing method,
        prepend the hash with ``algorithm:``, for example
        ``md5:pw9co2iun29juoh`` or ``sha1:092odwhi2ujdp2du2od2odh2wod2``.

    Returns
    -------
    action, verb : str
        The action that must be taken and the English verb (infinitive form of
        *action*) used in the log:
        * ``'download'``: File does not exist locally and must be downloaded.
        * ``'update'``: File exists locally but needs to be updated.
        * ``'fetch'``: File exists locally and only need to inform its path.


    """
    if not path.exists():
        action = 'download'
        verb = 'Downloading'
    elif not hash_matches(str(path), known_hash):
        action = 'update'
        verb = 'Updating'
    else:
        action = 'fetch'
        verb = 'Fetching'
    return (action, verb)