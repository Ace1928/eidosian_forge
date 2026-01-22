import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def file_specific_replacement(filepath, search_string, replace_string, strict=False):
    with openf(filepath, 'r+') as f:
        contents = f.read()
        if strict:
            contents = re.sub(f'\\b({re.escape(search_string)})\\b', lambda x: replace_string, contents)
        else:
            contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()