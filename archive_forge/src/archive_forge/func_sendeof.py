import os
import threading
import subprocess
import sys
import time
import signal
import shlex
from .spawnbase import SpawnBase, PY3
from .exceptions import EOF
from .utils import string_types
def sendeof(self):
    """Closes the stdin pipe from the writing end."""
    self.proc.stdin.close()