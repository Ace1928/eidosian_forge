from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union
from . import compat
def write_output_file_from_tempfile(self, tempfile: str, destination_path: str) -> None:
    if self.args.check:
        self._run_diff(destination_path, source_file=tempfile)
        os.unlink(tempfile)
    elif self.args.stdout:
        with open(tempfile) as tf:
            print(tf.read())
        os.unlink(tempfile)
    else:
        self.write_status(f'Writing {destination_path}...')
        shutil.move(tempfile, destination_path)
        self.write_status('done\n')