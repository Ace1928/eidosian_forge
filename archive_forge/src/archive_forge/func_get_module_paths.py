from __future__ import annotations
import collections
import os
import sys
import types
from pathlib import Path
from typing import Callable, Final
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
def get_module_paths(module: types.ModuleType) -> set[str]:
    paths_extractors = [lambda m: [m.__file__], lambda m: [m.__spec__.origin], lambda m: [p for p in m.__path__._path]]
    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            pass
        except Exception as e:
            _LOGGER.warning(f'Examining the path of {module.__name__} raised: {e}')
        all_paths.update([os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)])
    return all_paths