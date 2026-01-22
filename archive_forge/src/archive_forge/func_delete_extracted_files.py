import enum
import io
import os
import posixpath
import tarfile
import warnings
import zipfile
from datetime import datetime
from functools import partial
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import TrackedIterable, tracked_str
from .download_config import DownloadConfig
def delete_extracted_files(self):
    paths_to_delete = set(self.extracted_paths.values()) - set(self.downloaded_paths.values())
    for key, path in list(self.extracted_paths.items()):
        if path in paths_to_delete and os.path.isfile(path):
            os.remove(path)
            del self.extracted_paths[key]