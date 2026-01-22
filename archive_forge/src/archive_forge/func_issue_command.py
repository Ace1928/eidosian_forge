from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def issue_command(self, line):
    line = line.encode('utf-8') + b'\n'
    if self.verbose:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
    self.proc.stdin.write(line)
    self.proc.stdin.flush()