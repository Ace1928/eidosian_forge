from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def set_exec_mode(self, print_cmout: T.Optional[bool]=None, always_capture_stderr: T.Optional[bool]=None) -> None:
    if print_cmout is not None:
        self.print_cmout = print_cmout
    if always_capture_stderr is not None:
        self.always_capture_stderr = always_capture_stderr