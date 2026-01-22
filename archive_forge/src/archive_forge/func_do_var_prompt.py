from __future__ import annotations
import collections.abc as c
import codecs
import ctypes.util
import fcntl
import getpass
import io
import logging
import os
import random
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
import typing as t
from functools import wraps
from struct import unpack, pack
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsiblePromptInterrupt, AnsiblePromptNoninteractive
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import text_type
from ansible.utils.color import stringc
from ansible.utils.multiprocessing import context as multiprocessing_context
from ansible.utils.singleton import Singleton
from ansible.utils.unsafe_proxy import wrap_var
def do_var_prompt(self, varname: str, private: bool=True, prompt: str | None=None, encrypt: str | None=None, confirm: bool=False, salt_size: int | None=None, salt: str | None=None, default: str | None=None, unsafe: bool=False) -> str:
    result = None
    if sys.__stdin__.isatty():
        do_prompt = self.prompt
        if prompt and default is not None:
            msg = '%s [%s]: ' % (prompt, default)
        elif prompt:
            msg = '%s: ' % prompt
        else:
            msg = 'input for %s: ' % varname
        if confirm:
            while True:
                result = do_prompt(msg, private)
                second = do_prompt('confirm ' + msg, private)
                if result == second:
                    break
                self.display('***** VALUES ENTERED DO NOT MATCH ****')
        else:
            result = do_prompt(msg, private)
    else:
        result = None
        self.warning('Not prompting as we are not in interactive mode')
    if not result and default is not None:
        result = default
    if encrypt:
        from ansible.utils.encrypt import do_encrypt
        result = do_encrypt(result, encrypt, salt_size=salt_size, salt=salt)
    result = to_text(result, errors='surrogate_or_strict')
    if unsafe:
        result = wrap_var(result)
    return result