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
def execute_from_commandline(self, argv, cmd=None):
    argv = self._handle_reserved_options(argv)
    self.cmd = cmd if cmd is not None else self.cmd
    self.prog_name = os.path.basename(argv.pop(0))
    if not self.validate_arguments(argv):
        return self.error()
    return self.call_command(argv[0], argv[1:])