from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
@lru_cache()
def is_gdb_available():
    try:
        proc = subprocess.run(gdb_command + ['--version'], env=environment_for_gdb(), bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return False
    return proc.returncode == 0