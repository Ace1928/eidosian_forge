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
def verbose_git(cmd: T.List[str], workingdir: str, check: bool=False) -> bool:
    """
    Wrapper to convert GitException to WrapException caught in interpreter.
    """
    try:
        return mesonlib.verbose_git(cmd, workingdir, check=check)
    except mesonlib.GitException as e:
        raise WrapException(str(e))