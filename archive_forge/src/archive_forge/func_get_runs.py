import atexit
import datetime
import fnmatch
import os
import queue
import sys
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import quote as url_quote
import wandb
from wandb.proto import wandb_internal_pb2  # type: ignore
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context, datastore, handler, sender, tb_watcher
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.util import check_and_warn_old
def get_runs(include_offline: bool=True, include_online: bool=True, include_synced: bool=False, include_unsynced: bool=True, exclude_globs: Optional[List[str]]=None, include_globs: Optional[List[str]]=None):
    base = '.wandb' if os.path.exists('.wandb') else 'wandb'
    if not os.path.exists(base):
        return ()
    all_dirs = os.listdir(base)
    dirs = []
    if include_offline:
        dirs += filter(lambda _d: _d.startswith('offline-run-'), all_dirs)
    if include_online:
        dirs += filter(lambda _d: _d.startswith('run-'), all_dirs)
    fnames = []
    dirs.sort()
    for d in dirs:
        paths = os.listdir(os.path.join(base, d))
        if exclude_globs:
            paths = set(paths)
            for g in exclude_globs:
                paths = paths - set(fnmatch.filter(paths, g))
            paths = list(paths)
        if include_globs:
            new_paths = set()
            for g in include_globs:
                new_paths = new_paths.union(fnmatch.filter(paths, g))
            paths = list(new_paths)
        for f in paths:
            if f.endswith(WANDB_SUFFIX):
                fnames.append(os.path.join(base, d, f))
    filtered = []
    for f in fnames:
        dname = os.path.dirname(f)
        if os.path.exists(f'{f}{SYNCED_SUFFIX}') or os.path.basename(dname).startswith('run-'):
            if include_synced:
                filtered.append(_LocalRun(dname, True))
        elif include_unsynced:
            filtered.append(_LocalRun(dname, False))
    return tuple(filtered)