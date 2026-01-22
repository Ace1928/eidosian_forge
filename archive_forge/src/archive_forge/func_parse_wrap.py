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
def parse_wrap(self) -> None:
    try:
        config = configparser.ConfigParser(interpolation=None)
        config.read(self.filename, encoding='utf-8')
    except configparser.Error as e:
        raise WrapException(f'Failed to parse {self.basename}: {e!s}')
    self.parse_wrap_section(config)
    if self.type == 'redirect':
        dirname = Path(self.filename).parent
        fname = Path(self.values['filename'])
        for i, p in enumerate(fname.parts):
            if i % 2 == 0:
                if p == '..':
                    raise WrapException('wrap-redirect filename cannot contain ".."')
            elif p != 'subprojects':
                raise WrapException('wrap-redirect filename must be in the form foo/subprojects/bar.wrap')
        if fname.suffix != '.wrap':
            raise WrapException('wrap-redirect filename must be a .wrap file')
        fname = dirname / fname
        if not fname.is_file():
            raise WrapException(f'wrap-redirect {fname} filename does not exist')
        self.filename = str(fname)
        self.parse_wrap()
        self.redirected = True
    else:
        self.parse_provide_section(config)
    if 'patch_directory' in self.values:
        FeatureNew('Wrap files with patch_directory', '0.55.0').use(self.subproject)
    for what in ['patch', 'source']:
        if f'{what}_filename' in self.values and f'{what}_url' not in self.values:
            FeatureNew(f'Local wrap patch files without {what}_url', '0.55.0').use(self.subproject)