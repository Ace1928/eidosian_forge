import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
Add more ignore patterns to the ignore file in a tree.
    If ignore file does not exist then it will be created.
    The ignore file will be automatically added under version control.

    :param tree: Working tree to update the ignore list.
    :param name_pattern_list: List of ignore patterns.
    :return: None
    