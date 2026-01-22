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
def logger_class(self):
    uri = self.settings['logger_class'].get()
    if uri == 'simple':
        uri = LoggerClass.default
    if uri == LoggerClass.default:
        if 'statsd_host' in self.settings and self.settings['statsd_host'].value is not None:
            uri = 'gunicorn.instrument.statsd.Statsd'
    logger_class = util.load_class(uri, default='gunicorn.glogging.Logger', section='gunicorn.loggers')
    if hasattr(logger_class, 'install'):
        logger_class.install()
    return logger_class