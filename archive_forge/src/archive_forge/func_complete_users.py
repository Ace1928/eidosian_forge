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
def complete_users() -> List[str]:
    users = []
    if sys.platform.startswith('win'):
        expanded_path = os.path.expanduser(text)
        if os.path.isdir(expanded_path):
            user = text
            if add_trailing_sep_if_dir:
                user += os.path.sep
            users.append(user)
    else:
        import pwd
        for cur_pw in pwd.getpwall():
            if os.path.isdir(cur_pw.pw_dir):
                cur_user = '~' + cur_pw.pw_name
                if cur_user.startswith(text):
                    if add_trailing_sep_if_dir:
                        cur_user += os.path.sep
                    users.append(cur_user)
    if users:
        self.allow_appended_space = False
        self.allow_closing_quote = False
    return users