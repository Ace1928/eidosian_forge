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
def compile_example(verbose: bool=False) -> None:
    """
    Compile the example model.
    The current directory must be a cmdstan installation, i.e.,
    contains the makefile, Stanc compiler, and all libraries.

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    path = Path('examples', 'bernoulli', 'bernoulli').with_suffix(EXTENSION)
    if path.is_file():
        path.unlink()
    cmd = [MAKE, path.as_posix()]
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)
    except RuntimeError as e:
        raise CmdStanInstallError(f'Command "{' '.join(cmd)}" failed:\n{e}')
    if not path.is_file():
        raise CmdStanInstallError('Failed to generate example binary')