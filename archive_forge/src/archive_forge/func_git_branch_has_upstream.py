from __future__ import annotations
from dataclasses import dataclass, InitVar
import os, subprocess
import argparse
import asyncio
import threading
import copy
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import typing as T
import tarfile
import zipfile
from . import mlog
from .ast import IntrospectionInterpreter
from .mesonlib import quiet_git, GitException, Popen_safe, MesonException, windows_proof_rmtree
from .wrap.wrap import (Resolver, WrapException, ALL_TYPES,
def git_branch_has_upstream(self, urls: set) -> bool:
    cmd = ['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}']
    ret, upstream = quiet_git(cmd, self.repo_dir)
    if not ret:
        return False
    try:
        remote = upstream.split('/', maxsplit=1)[0]
    except IndexError:
        return False
    cmd = ['remote', 'get-url', remote]
    ret, remote_url = quiet_git(cmd, self.repo_dir)
    return remote_url.strip() in urls