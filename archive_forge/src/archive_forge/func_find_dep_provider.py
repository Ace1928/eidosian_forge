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
def find_dep_provider(self, packagename: str) -> T.Tuple[T.Optional[str], T.Optional[str]]:
    packagename = packagename.lower()
    wrap = self.provided_deps.get(packagename)
    if wrap:
        dep_var = wrap.provided_deps.get(packagename)
        return (wrap.name, dep_var)
    wrap_name = self.wrapdb_provided_deps.get(packagename)
    return (wrap_name, None)