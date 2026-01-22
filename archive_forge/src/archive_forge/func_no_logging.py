from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
@contextmanager
def no_logging(self) -> T.Iterator[None]:
    self.log_disable_stdout = True
    try:
        yield
    finally:
        self.log_disable_stdout = False