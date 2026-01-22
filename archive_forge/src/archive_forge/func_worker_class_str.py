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
def worker_class_str(self):
    uri = self.settings['worker_class'].get()
    is_sync = uri.endswith('SyncWorker') or uri == 'sync'
    if is_sync and self.threads > 1:
        return 'gthread'
    return uri