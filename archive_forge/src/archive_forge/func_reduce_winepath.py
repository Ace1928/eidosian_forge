from __future__ import annotations
import os, subprocess
import argparse
import tempfile
import shutil
import itertools
import typing as T
from pathlib import Path
from . import build, minstall
from .mesonlib import (EnvironmentVariables, MesonException, is_windows, setup_vsenv, OptionKey,
from . import mlog
def reduce_winepath(env: T.Dict[str, str]) -> None:
    winepath = env.get('WINEPATH')
    if not winepath:
        return
    winecmd = shutil.which('wine64') or shutil.which('wine')
    if not winecmd:
        return
    env['WINEPATH'] = get_wine_shortpath([winecmd], winepath.split(';'))
    mlog.log('Meson detected wine and has set WINEPATH accordingly')