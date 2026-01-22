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
def banner_cowsay(self, msg: str, color: str | None=None) -> None:
    if u': [' in msg:
        msg = msg.replace(u'[', u'')
        if msg.endswith(u']'):
            msg = msg[:-1]
    runcmd = [self.b_cowsay, b'-W', b'60']
    if self.noncow:
        thecow = self.noncow
        if thecow == 'random':
            thecow = random.choice(list(self.cows_available))
        runcmd.append(b'-f')
        runcmd.append(to_bytes(thecow))
    runcmd.append(to_bytes(msg))
    cmd = subprocess.Popen(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = cmd.communicate()
    self.display(u'%s\n' % to_text(out), color=color)