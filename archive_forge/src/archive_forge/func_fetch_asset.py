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
def fetch_asset(project_path: Path, url: str, dest: Path, checksum: Optional[str]=None) -> None:
    """Fetch an asset from a given URL or path. If a checksum is provided and a
    local file exists, it's only re-downloaded if the checksum doesn't match.

    project_path (Path): Path to project directory.
    url (str): URL or path to asset.
    checksum (Optional[str]): Optional expected checksum of local file.
    RETURNS (Optional[Path]): The path to the fetched asset or None if fetching
        the asset failed.
    """
    dest_path = (project_path / dest).resolve()
    if dest_path.exists():
        if checksum:
            if checksum == get_checksum(dest_path):
                msg.good(f'Skipping download with matching checksum: {dest}')
                return
        elif os.path.getsize(dest_path) == 0:
            msg.warn(f'Asset exists but with size of 0 bytes, deleting: {dest}')
            os.remove(dest_path)
    if not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True)
    with working_dir(project_path):
        url = convert_asset_url(url)
        try:
            download_file(url, dest_path)
            msg.good(f'Downloaded asset {dest}')
        except requests.exceptions.RequestException as e:
            if Path(url).exists() and Path(url).is_file():
                shutil.copy(url, str(dest_path))
                msg.good(f'Copied local asset {dest}')
            else:
                msg.fail(f'Download failed: {dest}', e)
    if checksum and checksum != get_checksum(dest_path):
        msg.fail(f"Checksum doesn't match value defined in {PROJECT_FILE}: {dest}")