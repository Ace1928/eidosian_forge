import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import typer
from wasabi import msg
from ..util import SimpleFrozenDict, download_file, ensure_path, get_checksum
from ..util import get_git_version, git_checkout, load_project_config
from ..util import parse_config_overrides, working_dir
from .main import PROJECT_FILE, Arg, Opt, app
def check_private_asset(dest: Path, checksum: Optional[str]=None) -> None:
    """Check and validate assets without a URL (private assets that the user
    has to provide themselves) and give feedback about the checksum.

    dest (Path): Destination path of the asset.
    checksum (Optional[str]): Optional checksum of the expected file.
    """
    if not Path(dest).exists():
        err = f'No URL provided for asset. You need to add this file yourself: {dest}'
        msg.warn(err)
    elif not checksum:
        msg.good(f'Asset already exists: {dest}')
    elif checksum == get_checksum(dest):
        msg.good(f'Asset exists with matching checksum: {dest}')
    else:
        msg.fail(f'Asset available but with incorrect checksum: {dest}')