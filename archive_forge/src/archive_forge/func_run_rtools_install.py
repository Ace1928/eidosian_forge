import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import OrderedDict
from time import sleep
from typing import Any, Dict, List
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import pushd, validate_dir, wrap_url_progress_hook
def run_rtools_install(args: Dict[str, Any]) -> None:
    """Main."""
    if platform.system() not in {'Windows'}:
        raise NotImplementedError(f'Download for the C++ toolchain on the current platform has not been implemented: {platform.system()}')
    toolchain = get_toolchain_name()
    version = args['version']
    if version is None:
        version = latest_version()
    version = normalize_version(version)
    print("C++ toolchain '{}' version: {}".format(toolchain, version))
    url = get_url(version)
    if 'verbose' in args:
        verbose = args['verbose']
    install_dir = args['dir']
    if install_dir is None:
        install_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
    validate_dir(install_dir)
    print('Install directory: {}'.format(install_dir))
    if 'progress' in args:
        progress = args['progress']
    else:
        progress = False
    if platform.system() == 'Windows':
        silent = 'silent' in args
        if 'silent' not in args and version in ('4.0', '4', '40'):
            silent = False
    else:
        silent = False
    toolchain_folder = get_toolchain_version(toolchain, version)
    with pushd(install_dir):
        if is_installed(toolchain_folder, version):
            print('C++ toolchain {} already installed'.format(toolchain_folder))
        else:
            if os.path.exists(toolchain_folder):
                shutil.rmtree(toolchain_folder, ignore_errors=False)
            retrieve_toolchain(toolchain_folder + EXTENSION, url, progress=progress)
            install_version(toolchain_folder, toolchain_folder + EXTENSION, version, silent, verbose)
        if 'no-make' not in args and platform.system() == 'Windows' and (version in ('4.0', '4', '40')):
            if os.path.exists(os.path.join(toolchain_folder, 'mingw64', 'bin', 'mingw32-make.exe')):
                print('mingw32-make.exe already installed')
            else:
                install_mingw32_make(toolchain_folder, verbose)