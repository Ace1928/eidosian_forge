import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
@property
def dummy_data_folder(self):
    if self.config is not None:
        return os.path.join('dummy', self.config.name, self.version_name)
    return os.path.join('dummy', self.version_name)