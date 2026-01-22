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
def git_rebase(self, revision: str) -> bool:
    try:
        self.git_output(['-c', 'rebase.autoStash=true', 'rebase', 'FETCH_HEAD'])
    except GitException as e:
        self.git_output(['-c', 'rebase.autoStash=true', 'rebase', '--abort'])
        self.log('  -> Could not rebase', mlog.bold(self.repo_dir), 'onto', mlog.bold(revision), '-- aborted')
        self.log(mlog.red(e.output))
        self.log(mlog.red(str(e)))
        return False
    return True