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
Log variant that only prints a given message one time per meson invocation.

        This considers ansi decorated values by the values they wrap without
        regard for the AnsiDecorator itself.
        