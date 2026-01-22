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
def retrieve_version(version: str, progress: bool=True) -> None:
    """Download specified CmdStan version."""
    if version is None or version == '':
        raise ValueError('Argument "version" unspecified.')
    if 'git:' in version:
        tag = version.split(':')[1]
        tag_folder = version.replace(':', '-').replace('/', '_')
        print(f"Cloning CmdStan branch '{tag}' from stan-dev/cmdstan on GitHub")
        do_command(['git', 'clone', '--depth', '1', '--branch', tag, '--recursive', '--shallow-submodules', 'https://github.com/stan-dev/cmdstan.git', f'cmdstan-{tag_folder}'])
        return
    print('Downloading CmdStan version {}'.format(version))
    url = get_download_url(version)
    for i in range(6):
        try:
            if progress and progbar.allow_show_progress():
                progress_hook: Optional[Callable[[int, int, int], None]] = wrap_url_progress_hook()
            else:
                progress_hook = None
            file_tmp, _ = urllib.request.urlretrieve(url, filename=None, reporthook=progress_hook)
            break
        except urllib.error.HTTPError as e:
            raise CmdStanRetrieveError('HTTPError: {}\nVersion {} not available from github.com.'.format(e.code, version)) from e
        except urllib.error.URLError as e:
            print('Failed to download CmdStan version {} from github.com'.format(version))
            print(e)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            print('Version {} not available from github.com.'.format(version))
            raise CmdStanRetrieveError('Version {} not available from github.com.'.format(version)) from e
    print('Download successful, file: {}'.format(file_tmp))
    try:
        print('Extracting distribution')
        tar = tarfile.open(file_tmp)
        first = tar.next()
        if first is not None:
            top_dir = first.name
        cmdstan_dir = f'cmdstan-{version}'
        if top_dir != cmdstan_dir:
            raise CmdStanInstallError('tarfile should contain top-level dir {},but found dir {} instead.'.format(cmdstan_dir, top_dir))
        target = os.getcwd()
        if is_windows():
            target = '\\\\?\\{}'.format(target)
        if progress and progbar.allow_show_progress():
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), colour='blue', leave=False):
                tar.extract(member=member)
        else:
            tar.extractall()
    except Exception as e:
        raise CmdStanInstallError(f'Failed to unpack file {file_tmp}, error:\n\t{str(e)}') from e
    finally:
        tar.close()
    print(f'Unpacked download as {cmdstan_dir}')