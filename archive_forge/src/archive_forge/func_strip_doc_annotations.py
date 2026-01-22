import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def strip_doc_annotations(doc: str) -> str:
    """
    Strip annotations from a docstring leaving only the text description

    :param doc: documentation string
    """
    cmd_desc = ''
    found_first = False
    for doc_line in doc.splitlines():
        stripped_line = doc_line.strip()
        if stripped_line.startswith(':'):
            if found_first:
                break
        elif stripped_line:
            if found_first:
                cmd_desc += '\n'
            cmd_desc += stripped_line
            found_first = True
        elif found_first:
            break
    return cmd_desc