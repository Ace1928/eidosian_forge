from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def fix_darwin(fname: str, new_rpath: str, final_path: str, install_name_mappings: T.Dict[str, str]) -> None:
    try:
        rpaths = get_darwin_rpaths_to_remove(fname)
    except subprocess.CalledProcessError:
        return
    try:
        args = []
        if rpaths:
            for rp in OrderedSet(rpaths):
                args += ['-delete_rpath', rp]
            subprocess.check_call(['install_name_tool', fname] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args = []
        if new_rpath:
            args += ['-add_rpath', new_rpath]
        if fname.endswith('dylib'):
            args += ['-id', final_path]
        if install_name_mappings:
            for old, new in install_name_mappings.items():
                args += ['-change', old, new]
        if args:
            subprocess.check_call(['install_name_tool', fname] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as err:
        raise SystemExit(err)