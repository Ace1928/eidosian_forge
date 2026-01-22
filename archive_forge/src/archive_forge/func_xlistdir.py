import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def xlistdir(path: str, download_config: Optional[DownloadConfig]=None) -> List[str]:
    """Extend `os.listdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(path).split('::')
    if is_local_path(main_hop):
        return os.listdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split('::')
        fs, *_ = fsspec.get_fs_token_paths(path, storage_options=storage_options)
        inner_path = main_hop.split('://')[-1]
        if inner_path.strip('/') and (not fs.isdir(inner_path)):
            raise FileNotFoundError(f"Directory doesn't exist: {path}")
        paths = fs.listdir(inner_path, detail=False)
        return [os.path.basename(path.rstrip('/')) for path in paths]