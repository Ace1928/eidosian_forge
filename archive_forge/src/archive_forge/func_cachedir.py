from __future__ import annotations
import inspect
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from .base import TestBase
from .. import config
from ..assertions import eq_
from ... import util
@config.fixture(scope='class')
def cachedir(self):
    yield from self._cachedir()