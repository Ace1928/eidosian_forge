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
def mask_comments(string):
    in_comment = ''
    prev_c = ''
    new_string = ''
    for c in string:
        if in_comment == '':
            if c == '/' and prev_c == '/':
                in_comment = '//'
            elif c == '*' and prev_c == '/':
                in_comment = '/*'
            elif c == '"' and prev_c != '\\' and (prev_c != "'"):
                in_comment = '"'
        elif in_comment == '//':
            if c == '\r' or c == '\n':
                in_comment = ''
        elif in_comment == '/*':
            if c == '/' and prev_c == '*':
                in_comment = ''
        elif in_comment == '"':
            if c == '"' and prev_c != '\\':
                in_comment = ''
        prev_c = c
        if in_comment == '':
            new_string += c
        else:
            new_string += 'x'
    return new_string