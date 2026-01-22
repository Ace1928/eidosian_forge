import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
def run_install(args: Union[InteractiveSettings, InstallationSettings]) -> None:
    """
    Run a (potentially interactive) installation
    """
    validate_dir(args.dir)
    print('CmdStan install directory: {}'.format(args.dir))
    _ = args.progress
    _ = args.verbose
    if args.compiler:
        run_compiler_install(args.dir, args.verbose, args.progress)
    if 'git:' in args.version:
        tag = args.version.replace(':', '-').replace('/', '_')
        cmdstan_version = f'cmdstan-{tag}'
    else:
        cmdstan_version = f'cmdstan-{args.version}'
    with pushd(args.dir):
        already_installed = os.path.exists(cmdstan_version) and os.path.exists(os.path.join(cmdstan_version, 'examples', 'bernoulli', 'bernoulli' + EXTENSION))
        if not already_installed or args.overwrite:
            if is_version_available(args.version):
                print('Installing CmdStan version: {}'.format(args.version))
            else:
                raise ValueError(f'Version {args.version} cannot be downloaded. Connection to GitHub failed. Check firewall settings or ensure this version exists.')
            shutil.rmtree(cmdstan_version, ignore_errors=True)
            retrieve_version(args.version, args.progress)
            install_version(cmdstan_version=cmdstan_version, overwrite=already_installed and args.overwrite, verbose=args.verbose, progress=args.progress, cores=args.cores)
        else:
            print('CmdStan version {} already installed'.format(args.version))
        with pushd(cmdstan_version):
            print('Test model compilation')
            compile_example(args.verbose)