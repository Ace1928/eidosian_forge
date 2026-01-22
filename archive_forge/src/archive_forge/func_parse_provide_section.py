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
def parse_provide_section(self, config: configparser.ConfigParser) -> None:
    if config.has_section('provides'):
        raise WrapException('Unexpected "[provides]" section, did you mean "[provide]"?')
    if config.has_section('provide'):
        for k, v in config['provide'].items():
            if k == 'dependency_names':
                names_dict = {n.strip().lower(): None for n in v.split(',')}
                self.provided_deps.update(names_dict)
                continue
            if k == 'program_names':
                names_list = [n.strip() for n in v.split(',')]
                self.provided_programs += names_list
                continue
            if not v:
                m = f'Empty dependency variable name for {k!r} in {self.basename}. If the subproject uses meson.override_dependency() it can be added in the "dependency_names" special key.'
                raise WrapException(m)
            self.provided_deps[k] = v