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
def open_wrapdburl(urlstring: str, allow_insecure: bool=False, have_opt: bool=False) -> 'http.client.HTTPResponse':
    if have_opt:
        insecure_msg = '\n\n    To allow connecting anyway, pass `--allow-insecure`.'
    else:
        insecure_msg = ''
    url = whitelist_wrapdb(urlstring)
    if has_ssl:
        try:
            return T.cast('http.client.HTTPResponse', urllib.request.urlopen(urllib.parse.urlunparse(url), timeout=REQ_TIMEOUT))
        except urllib.error.URLError as excp:
            msg = f'WrapDB connection failed to {urlstring} with error {excp}.'
            if isinstance(excp.reason, ssl.SSLCertVerificationError):
                if allow_insecure:
                    mlog.warning(f'{msg}\n\n    Proceeding without authentication.')
                else:
                    raise WrapException(f'{msg}{insecure_msg}')
            else:
                raise WrapException(msg)
    elif not allow_insecure:
        raise WrapException(f'SSL module not available in {sys.executable}: Cannot contact the WrapDB.{insecure_msg}')
    else:
        mlog.warning(f'SSL module not available in {sys.executable}: WrapDB traffic not authenticated.', once=True)
    nossl_url = url._replace(scheme='http')
    try:
        return T.cast('http.client.HTTPResponse', urllib.request.urlopen(urllib.parse.urlunparse(nossl_url), timeout=REQ_TIMEOUT))
    except urllib.error.URLError as excp:
        raise WrapException(f'WrapDB connection failed to {urlstring} with error {excp}')