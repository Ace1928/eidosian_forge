import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
def get_imports(file_path: str) -> Tuple[str, str, str, str]:
    """Find whether we should import or clone additional files for a given processing script.
        And list the import.

    We allow:
    - library dependencies,
    - local dependencies and
    - external dependencies whose url is specified with a comment starting from "# From:' followed by the raw url to a file, an archive or a github repository.
        external dependencies will be downloaded (and extracted if needed in the dataset folder).
        We also add an `__init__.py` to each sub-folder of a downloaded folder so the user can import from them in the script.

    Note that only direct import in the dataset processing script will be handled
    We don't recursively explore the additional import to download further files.

    Example::

        import tensorflow
        import .c4_utils
        import .clicr.dataset-code.build_json_dataset  # From: https://raw.githubusercontent.com/clips/clicr/master/dataset-code/build_json_dataset
    """
    lines = []
    with open(file_path, encoding='utf-8') as f:
        lines.extend(f.readlines())
    logger.debug(f'Checking {file_path} for additional imports.')
    imports: List[Tuple[str, str, str, Optional[str]]] = []
    is_in_docstring = False
    for line in lines:
        docstr_start_match = re.findall('[\\s\\S]*?"""[\\s\\S]*?', line)
        if len(docstr_start_match) == 1:
            is_in_docstring = not is_in_docstring
        if is_in_docstring:
            continue
        match = re.match('^import\\s+(\\.?)([^\\s\\.]+)[^#\\r\\n]*(?:#\\s+From:\\s+)?([^\\r\\n]*)', line, flags=re.MULTILINE)
        if match is None:
            match = re.match('^from\\s+(\\.?)([^\\s\\.]+)(?:[^\\s]*)\\s+import\\s+[^#\\r\\n]*(?:#\\s+From:\\s+)?([^\\r\\n]*)', line, flags=re.MULTILINE)
            if match is None:
                continue
        if match.group(1):
            if any((imp[1] == match.group(2) for imp in imports)):
                continue
            if match.group(3):
                url_path = match.group(3)
                url_path, sub_directory = _convert_github_url(url_path)
                imports.append(('external', match.group(2), url_path, sub_directory))
            elif match.group(2):
                imports.append(('internal', match.group(2), match.group(2), None))
        elif match.group(3):
            url_path = match.group(3)
            imports.append(('library', match.group(2), url_path, None))
        else:
            imports.append(('library', match.group(2), match.group(2), None))
    return imports