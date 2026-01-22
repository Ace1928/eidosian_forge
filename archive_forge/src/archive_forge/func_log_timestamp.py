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
def log_timestamp(self, *args: TV_Loggable) -> None:
    if self.log_timestamp_start:
        self.log(*args)