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
def force_logging(self) -> T.Iterator[None]:
    restore = self.log_disable_stdout
    self.log_disable_stdout = False
    try:
        yield
    finally:
        self.log_disable_stdout = restore