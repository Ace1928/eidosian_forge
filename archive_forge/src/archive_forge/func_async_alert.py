import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def async_alert(self, alert_msg: str, new_prompt: Optional[str]=None) -> None:
    """
        Display an important message to the user while they are at a command line prompt.
        To the user it appears as if an alert message is printed above the prompt and their current input
        text and cursor location is left alone.

        IMPORTANT: This function will not print an alert unless it can acquire self.terminal_lock to ensure
                   a prompt is onscreen. Therefore, it is best to acquire the lock before calling this function
                   to guarantee the alert prints and to avoid raising a RuntimeError.

                   This function is only needed when you need to print an alert while the main thread is blocking
                   at the prompt. Therefore, this should never be called from the main thread. Doing so will
                   raise a RuntimeError.

        :param alert_msg: the message to display to the user
        :param new_prompt: If you also want to change the prompt that is displayed, then include it here.
                           See async_update_prompt() docstring for guidance on updating a prompt.
        :raises RuntimeError: if called from the main thread.
        :raises RuntimeError: if called while another thread holds `terminal_lock`
        """
    if threading.current_thread() is threading.main_thread():
        raise RuntimeError('async_alert should not be called from the main thread')
    if not (vt100_support and self.use_rawinput):
        return
    if self.terminal_lock.acquire(blocking=False):
        update_terminal = False
        if alert_msg:
            alert_msg += '\n'
            update_terminal = True
        if new_prompt is not None:
            self.prompt = new_prompt
        cur_onscreen_prompt = rl_get_prompt()
        new_onscreen_prompt = self.continuation_prompt if self._at_continuation_prompt else self.prompt
        if new_onscreen_prompt != cur_onscreen_prompt:
            update_terminal = True
        if update_terminal:
            import shutil
            terminal_str = ansi.async_alert_str(terminal_columns=shutil.get_terminal_size().columns, prompt=cur_onscreen_prompt, line=readline.get_line_buffer(), cursor_offset=rl_get_point(), alert_msg=alert_msg)
            if rl_type == RlType.GNU:
                sys.stderr.write(terminal_str)
                sys.stderr.flush()
            elif rl_type == RlType.PYREADLINE:
                readline.rl.mode.console.write(terminal_str)
            rl_set_prompt(new_onscreen_prompt)
            rl_force_redisplay()
        self.terminal_lock.release()
    else:
        raise RuntimeError('another thread holds terminal_lock')