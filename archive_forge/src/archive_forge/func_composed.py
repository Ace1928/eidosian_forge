from __future__ import annotations
import contextlib
import functools
import operator
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from typing import Iterator
def composed(*args, **kwargs):
    with inner(*args, **kwargs) as saved, outer(saved) as res:
        yield res