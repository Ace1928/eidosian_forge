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
def local_path_to_dummy_data(self):
    return os.path.join(self.datasets_scripts_dir, self.dataset_name, self.dummy_zip_file)