import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
def setup_terminal(self, stdout, stderr, nosplash=False, quiet=False, verbose=False, no_color=False, **kwargs):
    self.stdout = stdout or sys.stdout
    self.stderr = stderr or sys.stderr
    self.nosplash = nosplash
    self.quiet = quiet
    self.verbose = verbose
    self.no_color = no_color