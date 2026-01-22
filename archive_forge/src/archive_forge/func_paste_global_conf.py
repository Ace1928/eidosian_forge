import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
@property
def paste_global_conf(self):
    raw_global_conf = self.settings['raw_paste_global_conf'].get()
    if raw_global_conf is None:
        return None
    global_conf = {}
    for e in raw_global_conf:
        s = util.bytes_to_str(e)
        try:
            k, v = re.split('(?<!\\\\)=', s, 1)
        except ValueError:
            raise RuntimeError('environment setting %r invalid' % s)
        k = k.replace('\\=', '=')
        v = v.replace('\\=', '=')
        global_conf[k] = v
    return global_conf