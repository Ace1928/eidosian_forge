import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def options_callback(group):
    group.add_argument('--twisted', '-T', action='store_true', help=_('Run twisted reactor.'))
    group.add_argument('--reactor', '-r', help=_('Select specific reactor (see --help-reactors). Implies --twisted.'))
    group.add_argument('--help-reactors', action='store_true', help=_('List available reactors for -r.'))
    group.add_argument('--plugin', '-p', help=_('twistd plugin to run (use twistd for a list). Use "--" to pass further options to the plugin.'))
    group.add_argument('--server', '-s', type=int, help=_('Port to run an eval server on (forces Twisted).'))