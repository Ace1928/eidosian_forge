import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@property
def summary(self) -> Union[str, bytes]:
    """:return: First line of the commit message"""
    if isinstance(self.message, str):
        return self.message.split('\n', 1)[0]
    else:
        return self.message.split(b'\n', 1)[0]