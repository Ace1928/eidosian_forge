from __future__ import annotations
import os
import sys
import shutil
import pickle
import typing as T
def rmtrees(build_dir: str, trees: T.List[str]) -> None:
    for t in trees:
        if os.path.isabs(t):
            print(f'Cannot delete dir with absolute path {t!r}')
            continue
        bt = os.path.join(build_dir, t)
        if os.path.isdir(bt):
            shutil.rmtree(bt, ignore_errors=True)