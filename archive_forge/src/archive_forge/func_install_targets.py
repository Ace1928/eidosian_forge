from __future__ import annotations
from glob import glob
import argparse
import errno
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import typing as T
import re
from . import build, environment
from .backend.backends import InstallData
from .mesonlib import (MesonException, Popen_safe, RealPathAction, is_windows,
from .scripts import depfixer, destdir_join
from .scripts.meson_exe import run_exe
def install_targets(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for t in d.targets:
        if is_aix():
            if not os.path.exists(t.fname) and '.so' in t.fname:
                t.fname = re.sub('[.][a]([.]?([0-9]+))*([.]?([a-z]+))*', '.a', t.fname.replace('.so', '.a'))
        if not self.should_install(t):
            continue
        if not os.path.exists(t.fname):
            if t.optional:
                self.log(f'File {t.fname!r} not found, skipping')
                continue
            else:
                raise MesonException(f'File {t.fname!r} could not be found')
        file_copied = False
        fname = check_for_stampfile(t.fname)
        outdir = get_destdir_path(destdir, fullprefix, t.outdir)
        outname = os.path.join(outdir, os.path.basename(fname))
        final_path = os.path.join(d.prefix, t.outdir, os.path.basename(fname))
        should_strip = t.strip or (t.can_strip and self.options.strip)
        install_rpath = t.install_rpath
        install_name_mappings = t.install_name_mappings
        install_mode = t.install_mode
        if not os.path.exists(fname):
            raise MesonException(f'File {fname!r} could not be found')
        elif os.path.isfile(fname):
            file_copied = self.do_copyfile(fname, outname, makedirs=(dm, outdir))
            if should_strip and d.strip_bin is not None:
                if fname.endswith('.jar'):
                    self.log('Not stripping jar target: {}'.format(os.path.basename(fname)))
                    continue
                self.do_strip(d.strip_bin, fname, outname)
            if fname.endswith('.js'):
                wasm_source = os.path.splitext(fname)[0] + '.wasm'
                if os.path.exists(wasm_source):
                    wasm_output = os.path.splitext(outname)[0] + '.wasm'
                    file_copied = self.do_copyfile(wasm_source, wasm_output)
        elif os.path.isdir(fname):
            fname = os.path.join(d.build_dir, fname.rstrip('/'))
            outname = os.path.join(outdir, os.path.basename(fname))
            dm.makedirs(outdir, exist_ok=True)
            self.do_copydir(d, fname, outname, None, install_mode, dm)
        else:
            raise RuntimeError(f'Unknown file type for {fname!r}')
        if file_copied:
            self.did_install_something = True
            try:
                self.fix_rpath(outname, t.rpath_dirs_to_remove, install_rpath, final_path, install_name_mappings, verbose=False)
            except SystemExit as e:
                if isinstance(e.code, int) and e.code == 0:
                    pass
                else:
                    raise
            self.set_mode(outname, install_mode, d.install_umask)