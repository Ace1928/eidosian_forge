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
def get_toolchain_name() -> str:
    """Return toolchain name."""
    if platform.system() == 'Windows':
        return 'RTools'
    return ''