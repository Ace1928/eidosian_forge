from __future__ import annotations
from .. import mlog
import contextlib
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import os
import hashlib
import shutil
import tempfile
import stat
import subprocess
import sys
import configparser
import time
import typing as T
import textwrap
import json
from base64 import b64encode
from netrc import netrc
from pathlib import Path, PurePath
from functools import lru_cache
from . import WrapMode
from .. import coredata
from ..mesonlib import quiet_git, GIT, ProgressBar, MesonException, windows_proof_rmtree, Popen_safe
from ..interpreterbase import FeatureNew
from ..interpreterbase import SubProject
from .. import mesonlib
def get_data_with_backoff(self, urlstring: str) -> T.Tuple[str, str]:
    delays = [1, 2, 4, 8, 16]
    for d in delays:
        try:
            return self.get_data(urlstring)
        except Exception as e:
            mlog.warning(f'failed to download with error: {e}. Trying after a delay...', fatal=False)
            time.sleep(d)
    return self.get_data(urlstring)