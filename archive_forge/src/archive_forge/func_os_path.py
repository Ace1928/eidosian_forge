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
def os_path(canonicalpath: str) -> str:
    return canonicalpath.replace(SEP, path.sep)