import contextlib
import filecmp
import os
import re
import shutil
import sys
import unicodedata
from io import StringIO
from os import path
from typing import Any, Generator, Iterator, List, Optional, Type
def make_filename_from_project(project: str) -> str:
    return make_filename(project_suffix_re.sub('', project)).lower()