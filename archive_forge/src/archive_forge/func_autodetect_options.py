from __future__ import annotations
from pathlib import Path
from enum import Enum
import subprocess
import shutil
import sys
import os
import re
from glob import glob
import typing as T
from mesonbuild import build, mesonlib, mlog
from mesonbuild.coredata import FORBIDDEN_TARGET_NAMES
from mesonbuild.environment import detect_ninja
from mesonbuild.templates.mesontemplates import create_meson_build
from mesonbuild.templates.samplefactory import sample_generator
def autodetect_options(options: Arguments, sample: bool=False) -> None:
    """
    Here we autodetect options for args not passed in so don't have to
    think about it.
    """
    if not options.name:
        options.name = Path().resolve().stem
        if not re.match('[a-zA-Z_][a-zA-Z0-9]*', options.name) and sample:
            raise SystemExit(f'Name of current directory "{options.name}" is not usable as a sample project name.\nSpecify a project name with --name.')
        print(f'Using "{options.name}" (name of current directory) as project name.')
    if not options.executable:
        options.executable = options.name
        print(f'Using "{options.executable}" (project name) as name of executable to build.')
    if options.executable in FORBIDDEN_TARGET_NAMES:
        raise mesonlib.MesonException(f'Executable name {options.executable!r} is reserved for Meson internal use. Refusing to init an invalid project.')
    if sample:
        return
    if not options.srcfiles:
        srcfiles: T.List[Path] = []
        for f in (f for f in Path().iterdir() if f.is_file()):
            if f.suffix in LANG_SUFFIXES:
                srcfiles.append(f)
        if not srcfiles:
            raise SystemExit('No recognizable source files found.\nRun meson init in an empty directory to create a sample project.')
        options.srcfiles = srcfiles
        print('Detected source files: ' + ' '.join((str(s) for s in srcfiles)))
    if not options.language:
        for f in options.srcfiles:
            if f.suffix == '.c':
                options.language = 'c'
                break
            if f.suffix in {'.cc', '.cpp'}:
                options.language = 'cpp'
                break
            if f.suffix == '.cs':
                options.language = 'cs'
                break
            if f.suffix == '.cu':
                options.language = 'cuda'
                break
            if f.suffix == '.d':
                options.language = 'd'
                break
            if f.suffix in FORTRAN_SUFFIXES:
                options.language = 'fortran'
                break
            if f.suffix == '.rs':
                options.language = 'rust'
                break
            if f.suffix == '.m':
                options.language = 'objc'
                break
            if f.suffix == '.mm':
                options.language = 'objcpp'
                break
            if f.suffix == '.java':
                options.language = 'java'
                break
            if f.suffix == '.vala':
                options.language = 'vala'
                break
        if not options.language:
            raise SystemExit("Can't autodetect language, please specify it with -l.")
        print('Detected language: ' + options.language)