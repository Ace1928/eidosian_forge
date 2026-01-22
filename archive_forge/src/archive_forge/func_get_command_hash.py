import hashlib
import os
import site
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from wasabi import msg
from ..errors import Errors
from ..util import check_spacy_env_vars, download_file, ensure_pathy, get_checksum
from ..util import get_hash, make_tempdir, upload_file
def get_command_hash(site_hash: str, env_hash: str, deps: List[Path], cmd: List[str]) -> str:
    """Create a hash representing the execution of a command. This includes the
    currently installed packages, whatever environment variables have been marked
    as relevant, and the command.
    """
    check_spacy_env_vars()
    dep_checksums = [get_checksum(dep) for dep in sorted(deps)]
    hashes = [site_hash, env_hash] + dep_checksums
    hashes.extend(cmd)
    creation_bytes = ''.join(hashes).encode('utf8')
    return hashlib.md5(creation_bytes).hexdigest()