from __future__ import annotations
import os
import sys
import json
import shutil
import tarfile
import platform
import subprocess
from typing import TYPE_CHECKING, List
from pathlib import Path
from argparse import ArgumentParser
import httpx
from .._errors import CLIError, SilentCLIError
from .._models import BaseModel
def grit(args: GritArgs) -> None:
    grit_path = install()
    try:
        subprocess.check_call([grit_path, *args.unknown_args])
    except subprocess.CalledProcessError:
        raise SilentCLIError() from None