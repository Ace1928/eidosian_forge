from __future__ import annotations
import os
import argparse
import subprocess
import tempfile
import shlex
import shutil
import typing as T
def run_join(build_dir: str, itstool: str, its_files: T.List[str], mo_files: T.List[str], in_fname: str, out_fname: str) -> int:
    if not mo_files:
        print('No mo files specified to use for translation.')
        return 1
    with tempfile.TemporaryDirectory(prefix=os.path.basename(in_fname), dir=build_dir) as tmp_dir:
        locale_mo_files = []
        for mo_file in mo_files:
            if not os.path.exists(mo_file):
                print(f'Could not find mo file {mo_file}')
                return 1
            if not mo_file.endswith('.mo'):
                print(f'File is not a mo file: {mo_file}')
                return 1
            parts = mo_file.partition('LC_MESSAGES')
            if parts[0].endswith((os.sep, '/')):
                locale = os.path.basename(parts[0][:-1])
            else:
                locale = os.path.basename(parts[0])
            tmp_mo_fname = os.path.join(tmp_dir, locale + '.mo')
            shutil.copy(mo_file, tmp_mo_fname)
            locale_mo_files.append(tmp_mo_fname)
        cmd = shlex.split(itstool)
        if its_files:
            for fname in its_files:
                cmd.extend(['-i', fname])
        cmd.extend(['-j', in_fname, '-o', out_fname])
        cmd.extend(locale_mo_files)
        return subprocess.call(cmd)