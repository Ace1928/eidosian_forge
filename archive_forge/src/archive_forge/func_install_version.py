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
def install_version(installation_dir: str, installation_file: str, version: str, silent: bool, verbose: bool=False) -> None:
    """Install specified toolchain version."""
    with pushd('.'):
        print('Installing the C++ toolchain: {}'.format(os.path.splitext(installation_file)[0]))
        cmd = [installation_file]
        cmd.extend(get_config(installation_dir, silent))
        print(' '.join(cmd))
        proc = subprocess.Popen(cmd, cwd=None, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
        while proc.poll() is None:
            if proc.stdout:
                output = proc.stdout.readline().decode('utf-8').strip()
                if output and verbose:
                    print(output, flush=True)
        _, stderr = proc.communicate()
        if proc.returncode:
            print('Installation failed: returncode={}'.format(proc.returncode))
            if stderr:
                print(stderr.decode('utf-8').strip())
            if is_installed(installation_dir, version):
                print('Installation files found at the installation location.')
            sys.exit(3)
    if is_installed(installation_dir, version):
        os.remove(installation_file)
    print('Installed {}'.format(os.path.splitext(installation_file)[0]))