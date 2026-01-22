import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
def check_glob(path, group, disable_glob):
    """Find files with glob."""
    if isinstance(path, str) and (not disable_glob):
        path_glob = glob(path)
        if path_glob:
            path = sorted(path_glob)
            msg = '\n'.join((f'{i}: {os.path.normpath(fpath)}' for i, fpath in enumerate(path, 1)))
            len_p = len(path)
            _log.info("glob found %d files for '%s':\n%s", len_p, group, msg)
    return path