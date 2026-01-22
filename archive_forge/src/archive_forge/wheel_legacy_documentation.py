import logging
import os.path
from typing import List, Optional
from pip._internal.cli.spinners import open_spinner
from pip._internal.utils.setuptools_build import make_setuptools_bdist_wheel_args
from pip._internal.utils.subprocess import call_subprocess, format_command_args
Build one unpacked package using the "legacy" build process.

    Returns path to wheel if successfully built. Otherwise, returns None.
    