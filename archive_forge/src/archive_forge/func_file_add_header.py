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
def file_add_header(filepath, header):
    with openf(filepath, 'r+') as f:
        contents = f.read()
        if header[0] != '<' and header[-1] != '>':
            header = f'"{header}"'
        contents = f'#include {header} \n' + contents
        f.seek(0)
        f.write(contents)
        f.truncate()